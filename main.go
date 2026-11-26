// 智能选餐 Web 应用：Go 标准库实现后端，调用本地 Ollama 大模型完成菜品推荐。
//
// 运行前提：
//  1. 已安装并启动 Ollama，且执行过 `ollama pull qwen2.5:3b`
//  2. 在本目录执行 `go run main.go`，浏览器访问 http://localhost:8080
package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// ---------------- 菜品库 ----------------

// Dish 描述一道菜品。json tag 同时暴露给 AI（拼进提示词）与前端（展示详情），
// 注意 ingredients（食材）字段名与提示词中“检查 ingredients”的约定保持一致。
type Dish struct {
	Name        string   `json:"name"`        // 菜名
	Tastes      []string `json:"tastes"`      // 口味标签，取值限于：酸 / 甜 / 苦 / 辣
	Ingredients []string `json:"ingredients"` // 主要食材（AI 判断“不要XX”的依据）
	Category    string   `json:"category"`    // 类别：荤菜 / 素菜 / 汤
}

// dishes 内置菜品库：19 道菜覆盖酸/甜/苦/辣四种口味与荤菜/素菜/汤三类。
// 其中多道菜含猪肉（麻婆豆腐、鱼香肉丝、红烧肉），便于验证“不要猪肉”类需求。
var dishes = []Dish{
	{Name: "麻婆豆腐", Tastes: []string{"辣"}, Ingredients: []string{"豆腐", "猪肉末", "豆瓣酱", "花椒"}, Category: "荤菜"},
	{Name: "辣子鸡丁", Tastes: []string{"辣"}, Ingredients: []string{"鸡肉", "干辣椒", "花椒", "花生"}, Category: "荤菜"},
	{Name: "水煮鱼", Tastes: []string{"辣"}, Ingredients: []string{"鱼", "豆芽", "豆瓣酱", "干辣椒"}, Category: "荤菜"},
	{Name: "宫保鸡丁", Tastes: []string{"辣", "甜"}, Ingredients: []string{"鸡肉", "花生", "干辣椒", "糖"}, Category: "荤菜"},
	{Name: "鱼香肉丝", Tastes: []string{"辣", "甜"}, Ingredients: []string{"猪肉", "木耳", "胡萝卜", "豆瓣酱"}, Category: "荤菜"},
	{Name: "酸菜鱼", Tastes: []string{"酸", "辣"}, Ingredients: []string{"鱼", "酸菜", "辣椒"}, Category: "荤菜"},
	{Name: "酸辣土豆丝", Tastes: []string{"酸", "辣"}, Ingredients: []string{"土豆", "醋", "干辣椒"}, Category: "素菜"},
	{Name: "糖醋排骨", Tastes: []string{"酸", "甜"}, Ingredients: []string{"排骨", "醋", "糖"}, Category: "荤菜"},
	{Name: "松鼠桂鱼", Tastes: []string{"酸", "甜"}, Ingredients: []string{"桂鱼", "糖", "醋", "番茄酱"}, Category: "荤菜"},
	{Name: "红烧肉", Tastes: []string{"甜"}, Ingredients: []string{"猪五花肉", "冰糖", "酱油"}, Category: "荤菜"},
	{Name: "可乐鸡翅", Tastes: []string{"甜"}, Ingredients: []string{"鸡翅", "可乐", "酱油"}, Category: "荤菜"},
	{Name: "拔丝地瓜", Tastes: []string{"甜"}, Ingredients: []string{"地瓜", "白糖"}, Category: "素菜"},
	{Name: "醋溜白菜", Tastes: []string{"酸"}, Ingredients: []string{"白菜", "醋", "干辣椒"}, Category: "素菜"},
	{Name: "西红柿鸡蛋汤", Tastes: []string{"酸"}, Ingredients: []string{"西红柿", "鸡蛋"}, Category: "汤"},
	{Name: "苦瓜炒蛋", Tastes: []string{"苦"}, Ingredients: []string{"苦瓜", "鸡蛋"}, Category: "素菜"},
	{Name: "凉拌苦瓜", Tastes: []string{"苦"}, Ingredients: []string{"苦瓜", "蒜", "香油"}, Category: "素菜"},
	{Name: "咖喱鸡肉", Tastes: []string{"苦", "辣"}, Ingredients: []string{"鸡肉", "咖喱块", "土豆", "洋葱"}, Category: "荤菜"},
	{Name: "酸辣汤", Tastes: []string{"酸", "辣"}, Ingredients: []string{"豆腐", "木耳", "鸡蛋", "胡椒粉"}, Category: "汤"},
	{Name: "苦瓜排骨汤", Tastes: []string{"苦"}, Ingredients: []string{"苦瓜", "排骨"}, Category: "汤"},
}

// ---------------- 配置（可用环境变量覆盖，便于调试） ----------------

func envDefault(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

var (
	ollamaURL   = envDefault("OLLAMA_URL", "http://localhost:11434") // Ollama 服务地址
	ollamaModel = envDefault("OLLAMA_MODEL", "qwen2.5:3b")           // 使用的本地模型
	listenAddr  = envDefault("LISTEN_ADDR", ":8080")                 // 后端监听地址
)

// ollamaClient 超时设为 120 秒：本地小模型冷启动加载 + 推理可能需要几十秒
var ollamaClient = &http.Client{Timeout: 120 * time.Second}

// ---------------- Ollama 调用 ----------------

// ollamaError 表示 Ollama 服务本身不可用（未启动、模型缺失、返回异常等）。
// 这类错误信息对用户有排查价值，会原样透传给前端；与“解析失败”区分开。
type ollamaError struct{ msg string }

func (e *ollamaError) Error() string { return e.msg }

// ollamaMessage 是 Ollama /api/chat 的消息结构。
// Role 取值：system / user / assistant / tool；assistant 消息可能携带 tool_calls。
type ollamaMessage struct {
	Role      string           `json:"role"`
	Content   string           `json:"content"`
	ToolCalls []ollamaToolCall `json:"tool_calls,omitempty"`
}

// ollamaToolCall 是模型发起的一次工具调用（arguments 为 JSON 对象）
type ollamaToolCall struct {
	Function struct {
		Name      string         `json:"name"`
		Arguments map[string]any `json:"arguments"`
	} `json:"function"`
}

// ollamaChatResponse 是 Ollama /api/chat 非流式响应中我们关心的字段
type ollamaChatResponse struct {
	Message ollamaMessage `json:"message"`
}

// ollamaChat 调用 Ollama /api/chat（携带工具定义，Function Calling）。
// onToken 非 nil 时走流式：模型文本增量实时上抛；工具调用从流式分块中收集
// （Ollama 会把完整 tool_calls 放在单个分块中返回）。
// 返回本轮助手消息——可能是最终文本回答，也可能携带 tool_calls 请求查库。
func ollamaChat(ctx context.Context, messages []ollamaMessage, onToken func(string)) (ollamaMessage, error) {
	payload, err := json.Marshal(map[string]any{
		"model":    ollamaModel,
		"messages": messages,
		"tools":    toolsSchema, // 工具定义：模型据此决定何时查询菜品库
		"stream":   onToken != nil,
		"options":  map[string]any{"temperature": 0.3},
	})
	if err != nil {
		return ollamaMessage{}, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, ollamaURL+"/api/chat", bytes.NewReader(payload))
	if err != nil {
		return ollamaMessage{}, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := ollamaClient.Do(req)
	if err != nil {
		return ollamaMessage{}, &ollamaError{msg: fmt.Sprintf(
			"无法连接本地 Ollama（%s）：请确认 Ollama 正在运行，且已执行 `ollama pull %s` 拉取模型",
			ollamaURL, ollamaModel)}
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 500))
		return ollamaMessage{}, &ollamaError{msg: fmt.Sprintf("Ollama 返回异常状态码 %d：%s",
			resp.StatusCode, strings.TrimSpace(string(body)))}
	}

	// 非流式：一次性解析完整回复
	if onToken == nil {
		var out ollamaChatResponse
		if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
			return ollamaMessage{}, fmt.Errorf("解析 Ollama 响应失败: %w", err)
		}
		return out.Message, nil
	}

	// 流式：逐行解析 NDJSON，累积文本增量与工具调用
	var content strings.Builder
	var toolCalls []ollamaToolCall
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1<<20) // 单行 JSON 可能较长，放大扫描缓冲上限
	for scanner.Scan() {
		var chunk struct {
			Message ollamaMessage `json:"message"`
			Done    bool          `json:"done"`
		}
		if err := json.Unmarshal(scanner.Bytes(), &chunk); err != nil {
			continue // 忽略空行等无法解析的杂项行
		}
		if chunk.Message.Content != "" {
			content.WriteString(chunk.Message.Content)
			onToken(chunk.Message.Content)
		}
		toolCalls = append(toolCalls, chunk.Message.ToolCalls...)
		if chunk.Done {
			break
		}
	}
	if err := scanner.Err(); err != nil {
		return ollamaMessage{}, err
	}
	return ollamaMessage{Role: "assistant", Content: content.String(), ToolCalls: toolCalls}, nil
}

// ---------------- 工具定义与实现（Function Calling） ----------------

// toolsSchema 提供给模型的工具定义。
// 菜品库不再整体塞进提示词：模型按需调用 searchDishes 查询，
// 因此菜品库可以无限扩大，而提示词长度与 Token 消耗保持稳定。
var toolsSchema = []map[string]any{
	{
		"type": "function",
		"function": map[string]any{
			"name":        "searchDishes",
			"description": "按条件搜索菜品库，返回匹配的菜品列表（含菜名、口味、食材、类别）",
			"parameters": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"taste": map[string]any{
						"type":        "string",
						"description": "想要的口味，取值：酸/甜/苦/辣；用户没提口味时传空字符串",
					},
					"exclude_ingredient": map[string]any{
						"type":        "string",
						"description": "需要排除的食材（如：猪肉）；用户没提忌口时传空字符串",
					},
					"category": map[string]any{
						"type":        "string",
						"description": "菜品类别，取值：荤菜/素菜/汤；用户没提类别时传空字符串",
					},
				},
				"required": []string{"taste", "exclude_ingredient", "category"},
			},
		},
	},
}

// buildSystemPrompt 构造系统提示词：介绍工具用法与最终输出格式要求。
func buildSystemPrompt(count int) string {
	return fmt.Sprintf(`你是一个智能点餐助手。菜品库规模很大，无法直接提供给你，你必须用工具查询。
可用的工具：searchDishes(taste, exclude_ingredient, category)
- taste：想要的口味，取值 酸/甜/苦/辣，用户没提就传空字符串 ""
- exclude_ingredient：需要排除的食材（如"猪肉"），用户没提就传空字符串 ""
- category：菜品类别，取值 荤菜/素菜/汤，用户没提就传空字符串 ""
- 工具返回匹配的菜品列表（含菜名、口味、食材、类别）

示例：用户说"不要猪肉，来一个汤"，应调用 searchDishes(taste="", exclude_ingredient="猪肉", category="汤")。

使用规则：
1. 你必须先调用 searchDishes 查询菜品库，绝不能凭空编造菜名。
2. 如果用户说"不要XX"，把 XX 作为 exclude_ingredient 传给工具。
3. 拿到查询结果后，从中挑选最符合用户需求的 %d 道菜（同时注意用户对类别的要求，如汤、荤菜、素菜）。
4. 如果查询结果中没有合适的菜，可以调整参数再次调用工具。
5. 最终回答只返回一个 JSON 数组，格式为 ["菜名1", "菜名2"]，不要输出任何其他解释文字。`, count)
}

// maxToolResults 工具单次最多返回的菜品条数：菜品库很大时防止把模型上下文撑爆
const maxToolResults = 50

// searchDishesResult 是 searchDishes 工具返回给模型的结果
type searchDishesResult struct {
	Total    int    `json:"total"`          // 符合条件的菜品总数
	Returned int    `json:"returned"`       // 实际返回条数（超过上限时被截断）
	Note     string `json:"note,omitempty"` // 给模型的提示（参数有误 / 结果被截断）
	Dishes   []Dish `json:"dishes"`         // 菜品列表
}

// validTastes 口味标签的合法取值
var validTastes = []string{"酸", "甜", "苦", "辣"}

// validCategories 菜品类别的合法取值
var validCategories = []string{"荤菜", "素菜", "汤"}

// containsString 判断列表中是否包含目标字符串
func containsString(list []string, target string) bool {
	for _, s := range list {
		if s == target {
			return true
		}
	}
	return false
}

// execSearchDishes 本地执行 searchDishes 工具（模型只发参数，不过模型的手）：
//  1. 校验并按 taste 过滤口味（非法口味返回提示，让模型自行修正）；
//  2. 校验并按 category 过滤类别（荤菜/素菜/汤）；
//  3. 按忌口过滤——用户输入解析出的关键词（forbidden）始终生效，并与模型传入的
//     exclude_ingredient 合并，保证“不要XX”这类硬约束不依赖模型自觉；
//  4. 超过 maxToolResults 时截断并附提示。
func execSearchDishes(taste, exclude, category string, forbidden []string) searchDishesResult {
	if taste != "" && !containsString(validTastes, taste) {
		return searchDishesResult{Note: fmt.Sprintf("口味参数仅支持 %v，请修正后重新调用", validTastes)}
	}
	if category != "" && !containsString(validCategories, category) {
		return searchDishesResult{Note: fmt.Sprintf("类别参数仅支持 %v，请修正后重新调用", validCategories)}
	}

	// 1) 按口味与类别过滤
	pool := make([]Dish, 0, len(dishes))
	for _, d := range dishes {
		if taste != "" && !containsString(d.Tastes, taste) {
			continue
		}
		if category != "" && d.Category != category {
			continue
		}
		pool = append(pool, d)
	}

	// 2) 按忌口过滤（用户忌口 + 模型传入的排除食材合并生效）
	excludes := append([]string{}, forbidden...)
	if exclude != "" {
		excludes = append(excludes, exclude)
	}
	filtered := filterByForbidden(pool, excludes)

	// 3) 截断
	res := searchDishesResult{Total: len(filtered), Dishes: filtered}
	if len(filtered) > maxToolResults {
		res.Dishes = filtered[:maxToolResults]
		res.Note = fmt.Sprintf("匹配菜品过多，仅返回前 %d 条，请增加条件缩小范围", maxToolResults)
	}
	res.Returned = len(res.Dishes)
	if res.Dishes == nil {
		res.Dishes = []Dish{}
	}
	return res
}

// ---------------- Agent 工具调用循环 ----------------

// maxToolRounds 工具调用对话的最大轮次，防止模型无限循环调用工具
const maxToolRounds = 4

// runAgentLoop 驱动“模型 <-> 工具”的多轮对话：
//  1. 发送系统提示词 + 用户需求；
//  2. 模型若返回 tool_calls，就在本地执行 searchDishes 并把结果以 role=tool 消息回传，进入下一轮；
//  3. 模型不再调用工具时的回复即为最终回答。
//
// onToken / onStatus 为流式回调（可为 nil）：onToken 实时上抛模型文本增量，
// onStatus 上抛过程状态（如“正在查询菜品库”）。
// 返回最终回答文本，以及 allowed：所有工具返回过的菜品集合——
// 最终菜名必须出自这里，既防编造，也保证忌口约束生效。
func runAgentLoop(ctx context.Context, plan *recommendPlan, onToken func(string), onStatus func(string)) (string, map[string]Dish, error) {
	messages := []ollamaMessage{
		{Role: "system", Content: plan.SystemPrompt},
		{Role: "user", Content: fmt.Sprintf("%s\n\n（用户需要 %d 道菜）", plan.Query, plan.Count)},
	}
	allowed := make(map[string]Dish)
	nudged := false

	for round := 1; round <= maxToolRounds; round++ {
		reply, err := ollamaChat(ctx, messages, onToken)
		if err != nil {
			return "", nil, err
		}
		messages = append(messages, reply)

		// 模型没有调用工具：菜品库不在上下文里，不允许它凭空回答
		if len(reply.ToolCalls) == 0 {
			if len(allowed) == 0 && !nudged {
				nudged = true
				log.Printf("模型未调用工具，发送提醒要求先查库")
				if onStatus != nil {
					onStatus("AI 没有先查询菜品库，已提醒它调用工具")
				}
				messages = append(messages, ollamaMessage{
					Role:    "user",
					Content: "请先调用 searchDishes 工具查询菜品库，再根据查询结果回答。",
				})
				continue
			}
			return reply.Content, allowed, nil
		}

		// 模型请求调用工具：本地执行，结果以 role=tool 消息回传
		for _, tc := range reply.ToolCalls {
			if tc.Function.Name != "searchDishes" {
				log.Printf("模型请求了未知工具 %q，忽略", tc.Function.Name)
				continue
			}
			taste, _ := tc.Function.Arguments["taste"].(string)
			exclude, _ := tc.Function.Arguments["exclude_ingredient"].(string)
			category, _ := tc.Function.Arguments["category"].(string)
			log.Printf("模型调用工具 searchDishes(taste=%q, exclude_ingredient=%q, category=%q)", taste, exclude, category)
			if onStatus != nil {
				onStatus(fmt.Sprintf("AI 正在查询菜品库：taste=%s，exclude_ingredient=%s，category=%s", taste, exclude, category))
			}
			result := execSearchDishes(taste, exclude, category, plan.Forbidden)
			if result.Note != "" {
				log.Printf("工具提示: %s", result.Note)
			}
			log.Printf("工具返回 %d/%d 道菜", result.Returned, result.Total)
			for _, d := range result.Dishes {
				allowed[d.Name] = d
			}
			resultJSON, _ := json.Marshal(result)
			messages = append(messages, ollamaMessage{Role: "tool", Content: string(resultJSON)})
		}
	}
	return "", nil, errors.New("工具调用轮次超限，AI 未能给出最终答案")
}

// ---------------- 数量解析 ----------------

// countRe 匹配“数量 + 量词”，如“2道”“两道菜”“三个”“来一份”
var countRe = regexp.MustCompile(`([0-9]+|[一两二三四五六七八九十]+)\s*[道个份样碟碗盘]`)

// defaultCount 用户未指明数量时默认推荐 3 道菜
const defaultCount = 3

// parseCount 从用户输入中解析需要的菜品数量。
// 支持阿拉伯数字与简单中文数字（一~九十九），无法识别时返回默认值。
func parseCount(query string) int {
	m := countRe.FindStringSubmatch(query)
	if m == nil {
		return defaultCount
	}
	n, err := strconv.Atoi(m[1])
	if err != nil {
		n = chineseToInt(m[1])
	}
	return clampCount(n)
}

// clampCount 把数量限制在 [defaultCount] 与菜品库规模之间：非法或 0 用默认值，超大则封顶
func clampCount(n int) int {
	if n <= 0 {
		return defaultCount
	}
	if n > len(dishes) {
		return len(dishes)
	}
	return n
}

// chineseToInt 将简单中文数字转为整数（“两”按 2 处理），
// 仅覆盖“一”到“九十九”的常见写法，无法解析返回 0。
func chineseToInt(s string) int {
	digit := func(r rune) int {
		switch r {
		case '一':
			return 1
		case '两', '二':
			return 2
		case '三':
			return 3
		case '四':
			return 4
		case '五':
			return 5
		case '六':
			return 6
		case '七':
			return 7
		case '八':
			return 8
		case '九':
			return 9
		}
		return 0
	}
	if strings.ContainsRune(s, '十') {
		parts := strings.SplitN(s, "十", 2)
		tens := 1 // “十”“十五”省略“几”时默认一十
		if r := []rune(parts[0]); len(r) > 0 {
			tens = digit(r[0])
		}
		ones := 0
		if len(parts) == 2 {
			if r := []rune(parts[1]); len(r) > 0 {
				ones = digit(r[0])
			}
		}
		return tens*10 + ones
	}
	runes := []rune(s)
	if len(runes) == 1 {
		return digit(runes[0])
	}
	return 0
}

// ---------------- 忌口解析 ----------------

// forbiddenRe 匹配“不要XX / 不吃XX / 别吃XX / 忌口XX”等忌口表达，捕获忌口内容（食材或口味）。
// 捕获组使用 [\p{Han}0-9a-zA-Z]+，在标点/空格处自然截断，
// 如“不要猪肉，来两道菜”捕获“猪肉”、“不要辣的”捕获“辣的”（后面再去掉“的”）。
var forbiddenRe = regexp.MustCompile(`(?:不要|不吃|别吃|别放|别加|忌口|忌|没有)([\p{Han}0-9a-zA-Z]+)`)

// parseForbidden 从用户输入中提取所有忌口关键词（如“不要猪肉”->“猪肉”）。
// 后端据此直接过滤菜品库，作为小模型可能无视提示词约束时的兜底保障。
func parseForbidden(query string) []string {
	matches := forbiddenRe.FindAllStringSubmatch(query, -1)
	var out []string
	for _, m := range matches {
		w := strings.TrimSuffix(m[1], "的") // “不要辣的”->“辣”
		if w != "" {
			out = append(out, w)
		}
	}
	return out
}

// filterByForbidden 按忌口关键词过滤菜品，命中任一规则即剔除：
//  1. 忌口词与菜名互相包含（“不要红烧肉”命中“红烧肉”）；
//  2. 忌口词与任一食材互相包含（“猪肉”命中“猪肉末”/“猪五花肉”，双向包含可兼容
//     用户写“不要猪肉末”而库里写“猪肉”的情况）；
//  3. 忌口词本身是口味标签（“不要辣”排除所有辣味菜）。
func filterByForbidden(menu []Dish, forbidden []string) []Dish {
	out := make([]Dish, 0, len(menu))
	for _, d := range menu {
		banned := false
		for _, f := range forbidden {
			if strings.Contains(d.Name, f) {
				banned = true
				break
			}
			for _, ing := range d.Ingredients {
				if strings.Contains(ing, f) || strings.Contains(f, ing) {
					banned = true
					break
				}
			}
			if !banned {
				for _, t := range d.Tastes {
					if t == f {
						banned = true
						break
					}
				}
			}
			if banned {
				break
			}
		}
		if !banned {
			out = append(out, d)
		}
	}
	return out
}

// ---------------- 模型输出解析 ----------------

// extractJSONArray 从模型输出文本中提取 JSON 字符串数组。
// 小模型常在数组前后附带“好的，以下是推荐：”等解释文字，
// 因此定位第一个 '[' 与最后一个 ']' 之间的内容再解析。
func extractJSONArray(text string) ([]string, error) {
	start := strings.Index(text, "[")
	end := strings.LastIndex(text, "]")
	if start < 0 || end <= start {
		return nil, errors.New("模型输出中未找到 JSON 数组")
	}
	var names []string
	if err := json.Unmarshal([]byte(text[start:end+1]), &names); err != nil {
		return nil, fmt.Errorf("JSON 数组解析失败: %w", err)
	}
	return names, nil
}

// selectDishes 解析模型最终回答并映射回“工具查过的菜品集合”：
//  1. 提取 JSON 数组；
//  2. 剔除不在 allowed 中的菜名（模型编造，或工具从未返回过）；
//  3. 按用户要的数量截断。
func selectDishes(text string, count int, allowed map[string]Dish) ([]Dish, error) {
	names, err := extractJSONArray(text)
	if err != nil {
		return nil, err
	}
	log.Printf("模型返回菜名: %q", names)

	seen := make(map[string]bool, len(names))
	var selected []Dish
	var unknownNames []string
	for _, n := range names {
		n = strings.TrimSpace(n)
		if d, ok := allowed[n]; ok && !seen[n] {
			seen[n] = true
			selected = append(selected, d)
		} else if !ok {
			unknownNames = append(unknownNames, n)
		}
	}
	if len(unknownNames) > 0 {
		log.Printf("已剔除不在查询结果中的菜名: %q", unknownNames)
	}
	if len(selected) == 0 {
		return nil, fmt.Errorf("模型返回的菜名均不在工具查询结果中: %v", names)
	}
	if len(selected) > count {
		selected = selected[:count]
	}
	return selected, nil
}

// ---------------- 请求准备（普通/流式接口共用） ----------------

// recommendPlan 一次推荐请求的准备工作产物
type recommendPlan struct {
	Query        string   // 原始用户需求（拼进 user 消息）
	Count        int      // 用户要几道菜
	Forbidden    []string // 识别到的忌口关键词（工具实现层强制生效）
	SystemPrompt string   // 系统提示词（介绍工具用法与输出要求）
}

// prepareRecommend 解析数量与忌口。
// 注意：菜品库不再放进提示词——模型将通过 searchDishes 工具按需查询；
// 这里只做忌口快速失败检查：整个菜品库都被忌口过滤掉时无需进入工具循环。
func prepareRecommend(query string) (*recommendPlan, error) {
	count := parseCount(query)
	forbidden := parseForbidden(query)
	if len(forbidden) > 0 {
		remaining := filterByForbidden(dishes, forbidden)
		log.Printf("识别到忌口: %q，可选菜品从 %d 道过滤为 %d 道", forbidden, len(dishes), len(remaining))
		if len(remaining) == 0 {
			log.Printf("忌口 %v 过滤后菜品库为空", forbidden)
			return nil, fmt.Errorf("菜品库中没有符合忌口 %v 的菜品，换个忌口试试？", forbidden)
		}
	}
	return &recommendPlan{
		Query:        query,
		Count:        count,
		Forbidden:    forbidden,
		SystemPrompt: buildSystemPrompt(count),
	}, nil
}

// logPrompt 打印完整模型输入日志（调试用，普通/流式接口共用）
func logPrompt(prompt string) {
	log.Printf("========== 模型输入 [模型: %s，%d 字符] ==========\n%s\n========== 模型输入结束 ==========",
		ollamaModel, len([]rune(prompt)), prompt)
}

// ---------------- HTTP 处理 ----------------

type recommendRequest struct {
	Query string `json:"query"` // 用户自然语言需求
}

type recommendResponse struct {
	Dishes []Dish `json:"dishes"` // 推荐菜品（连同口味/食材/类别一起返回，便于前端展示）
}

type errorResponse struct {
	Error string `json:"error"`
}

// indexHandler 返回前端页面 index.html（直接读磁盘文件，前端改动后刷新即可，无需重新编译）
func indexHandler(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	http.ServeFile(w, r, "index.html")
}

// recommendHandler 核心接口：用户需求 -> 构造 Prompt -> Ollama 生成 -> 解析校验 -> 返回菜品
func recommendHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, errorResponse{Error: "请使用 POST 请求"})
		return
	}

	var req recommendRequest
	if err := json.NewDecoder(io.LimitReader(r.Body, 4<<10)).Decode(&req); err != nil || strings.TrimSpace(req.Query) == "" {
		writeJSON(w, http.StatusBadRequest, errorResponse{Error: "请输入你的用餐需求，例如：想吃点辣的，不要猪肉，来两道菜"})
		return
	}
	query := strings.TrimSpace(req.Query)

	// 解析数量与忌口（流式接口复用同一套准备逻辑）
	plan, err := prepareRecommend(query)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, errorResponse{Error: err.Error()})
		return
	}
	log.Printf("收到需求: %q（需要 %d 道菜）", query, plan.Count)
	// 打印系统提示词（工具用法与输出要求），便于调试模型行为
	logPrompt(plan.SystemPrompt)

	// 驱动“模型 <-> searchDishes 工具”的多轮对话。
	// 小模型输出偶发不稳定，解析失败自动重试一次；连接类错误重试无意义，立即结束。
	var lastErr error
	for attempt := 1; attempt <= 2; attempt++ {
		start := time.Now()
		finalText, allowed, err := runAgentLoop(r.Context(), plan, nil, nil)
		if err != nil {
			lastErr = err
			log.Printf("Agent 循环失败（耗时 %v）: %v", time.Since(start), err)
			break
		}
		// 打印模型最终输出（不截断），便于定位解析失败的具体原因
		log.Printf("========== 模型输出 [耗时 %v] ==========\n%s\n========== 模型输出结束 ==========",
			time.Since(start), finalText)
		selected, err := selectDishes(finalText, plan.Count, allowed)
		if err == nil {
			log.Printf("推荐成功: %v", dishNames(selected))
			writeJSON(w, http.StatusOK, recommendResponse{Dishes: selected})
			return
		}
		log.Printf("第 %d 次解析失败: %v", attempt, err)
		lastErr = err
	}

	// 两种失败分别给出不同提示：服务不可用 -> 具体排查建议；生成/解析失败 -> 友好话术
	var ollErr *ollamaError
	if errors.As(lastErr, &ollErr) {
		log.Printf("推荐失败: %v", ollErr.msg)
		writeJSON(w, http.StatusServiceUnavailable, errorResponse{Error: ollErr.msg})
		return
	}
	log.Printf("推荐失败: %v", lastErr)
	writeJSON(w, http.StatusBadGateway, errorResponse{Error: "AI 暂时没想好，请换个说法试试"})
}

// ---------------- SSE 流式推荐 ----------------

// writeSSE 向客户端写出一条 SSE 事件并立即推送。
// 协议格式：event: <事件名>\ndata: <JSON>\n\n（JSON 永远压成单行，不破坏 SSE 分帧）
func writeSSE(w http.ResponseWriter, flusher http.Flusher, event string, payload any) {
	data, _ := json.Marshal(payload)
	fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, data)
	flusher.Flush()
}

// recommendStreamHandler 流式推荐接口（SSE），事件约定：
//   - token  —— 模型增量输出片段 {"text":"..."}，前端实时展示“AI 思考过程”
//   - status —— 过程状态 {"message":"..."}（如解析失败自动重试）
//   - result —— 最终推荐菜品，结构与普通接口一致 {"dishes":[...]}
//   - error  —— 失败信息 {"error":"..."}
//
// 前端用 fetch + ReadableStream 读取（原生 EventSource 只支持 GET，无法带 POST body）。
func recommendStreamHandler(w http.ResponseWriter, r *http.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeJSON(w, http.StatusInternalServerError, errorResponse{Error: "当前环境不支持流式响应"})
		return
	}
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, errorResponse{Error: "请使用 POST 请求"})
		return
	}

	// SSE 必需的响应头
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")

	var req recommendRequest
	if err := json.NewDecoder(io.LimitReader(r.Body, 4<<10)).Decode(&req); err != nil || strings.TrimSpace(req.Query) == "" {
		writeSSE(w, flusher, "error", errorResponse{Error: "请输入你的用餐需求，例如：想吃点辣的，不要猪肉，来两道菜"})
		return
	}
	query := strings.TrimSpace(req.Query)

	plan, err := prepareRecommend(query)
	if err != nil {
		writeSSE(w, flusher, "error", errorResponse{Error: err.Error()})
		return
	}
	log.Printf("收到需求(流式): %q（需要 %d 道菜）", query, plan.Count)
	logPrompt(plan.SystemPrompt)

	// 驱动“模型 <-> searchDishes 工具”的多轮对话：
	// 工具查询过程通过 status 事件实时告知前端，最终回答的文本增量通过 token 事件流式推送。
	for attempt := 1; attempt <= 2; attempt++ {
		if attempt > 1 {
			writeSSE(w, flusher, "status", map[string]string{"message": "输出解析失败，自动重试中…"})
		}
		start := time.Now()
		finalText, allowed, err := runAgentLoop(r.Context(), plan,
			func(tok string) {
				writeSSE(w, flusher, "token", map[string]string{"text": tok})
			},
			func(msg string) {
				writeSSE(w, flusher, "status", map[string]string{"message": msg})
			},
		)
		if err != nil {
			// 客户端主动断开（如关闭页面）时静默结束
			if r.Context().Err() != nil {
				log.Printf("客户端断开连接，停止本次流式推荐")
				return
			}
			log.Printf("Agent 循环失败（耗时 %v）: %v", time.Since(start), err)
			msg := "AI 暂时没想好，请换个说法试试"
			var ollErr *ollamaError
			if errors.As(err, &ollErr) {
				msg = ollErr.msg
			}
			writeSSE(w, flusher, "error", errorResponse{Error: msg})
			return
		}
		// 打印模型最终输出（不截断），便于定位解析失败的具体原因
		log.Printf("========== 模型输出 [耗时 %v] ==========\n%s\n========== 模型输出结束 ==========",
			time.Since(start), finalText)

		selected, err := selectDishes(finalText, plan.Count, allowed)
		if err == nil {
			log.Printf("推荐成功: %v", dishNames(selected))
			writeSSE(w, flusher, "result", recommendResponse{Dishes: selected})
			return
		}
		log.Printf("第 %d 次解析失败: %v", attempt, err)
	}
	log.Printf("推荐失败: 两次解析均失败")
	writeSSE(w, flusher, "error", errorResponse{Error: "AI 暂时没想好，请换个说法试试"})
}

// writeJSON 统一的 JSON 响应输出
func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

// dishNames 返回菜名列表（仅用于日志）
func dishNames(ds []Dish) []string {
	names := make([]string, len(ds))
	for i, d := range ds {
		names[i] = d.Name
	}
	return names
}

func main() {
	mux := http.NewServeMux()
	mux.HandleFunc("/", indexHandler)                               // GET / 返回前端页面
	mux.HandleFunc("/api/recommend", recommendHandler)              // POST 推荐接口（一次性返回）
	mux.HandleFunc("/api/recommend/stream", recommendStreamHandler) // POST 流式推荐接口（SSE）

	log.Printf("智能选餐已启动，访问 http://localhost%s （Ollama: %s，模型: %s）", listenAddr, ollamaURL, ollamaModel)
	if err := http.ListenAndServe(listenAddr, mux); err != nil {
		log.Fatalf("服务启动失败: %v", err)
	}
}
