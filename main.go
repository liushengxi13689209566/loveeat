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

// ollamaGenerateResponse 是 Ollama /api/generate 非流式响应中我们关心的字段
type ollamaGenerateResponse struct {
	Response string `json:"response"` // 模型生成的文本
}

// callOllama 调用本地 Ollama 的 /api/generate 接口（非流式），返回模型输出文本。
func callOllama(prompt string) (string, error) {
	// 构造请求体：stream=false 一次性返回完整结果；调低 temperature 让筛选结果更稳定
	payload, err := json.Marshal(map[string]any{
		"model":   ollamaModel,
		"prompt":  prompt,
		"stream":  false,
		"options": map[string]any{"temperature": 0.3},
	})
	if err != nil {
		return "", err
	}

	resp, err := ollamaClient.Post(ollamaURL+"/api/generate", "application/json", bytes.NewReader(payload))
	if err != nil {
		// 连接失败是最常见的问题：Ollama 未启动或模型未拉取，给出具体排查提示
		return "", &ollamaError{msg: fmt.Sprintf(
			"无法连接本地 Ollama（%s）：请确认 Ollama 正在运行，且已执行 `ollama pull %s` 拉取模型",
			ollamaURL, ollamaModel)}
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 500))
		return "", &ollamaError{msg: fmt.Sprintf("Ollama 返回异常状态码 %d：%s",
			resp.StatusCode, strings.TrimSpace(string(body)))}
	}

	var out ollamaGenerateResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return "", fmt.Errorf("解析 Ollama 响应失败: %w", err)
	}
	if strings.TrimSpace(out.Response) == "" {
		return "", errors.New("Ollama 返回了空内容")
	}
	return out.Response, nil
}

// ollamaStreamChunk 是 Ollama 流式响应中一行 NDJSON 的关键字段。
// stream=true 时 Ollama 按行返回生成进度，每行一个 JSON 对象，
// response 字段是本次新增的文本片段，最后一行 done=true。
type ollamaStreamChunk struct {
	Response string `json:"response"`
	Done     bool   `json:"done"`
}

// callOllamaStream 以流式方式调用 Ollama /api/generate（stream=true），
// 每收到一段增量文本就通过 onToken 回调上抛（供 SSE 实时转发给前端），
// 函数返回拼接后的完整输出文本。
// ctx 绑定到对 Ollama 的请求上：客户端（浏览器）断开连接时会同步中断生成。
func callOllamaStream(ctx context.Context, prompt string, onToken func(string)) (string, error) {
	payload, err := json.Marshal(map[string]any{
		"model":   ollamaModel,
		"prompt":  prompt,
		"stream":  true, // 流式：Ollama 逐段返回生成内容
		"options": map[string]any{"temperature": 0.3},
	})
	if err != nil {
		return "", err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, ollamaURL+"/api/generate", bytes.NewReader(payload))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := ollamaClient.Do(req)
	if err != nil {
		return "", &ollamaError{msg: fmt.Sprintf(
			"无法连接本地 Ollama（%s）：请确认 Ollama 正在运行，且已执行 `ollama pull %s` 拉取模型",
			ollamaURL, ollamaModel)}
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 500))
		return "", &ollamaError{msg: fmt.Sprintf("Ollama 返回异常状态码 %d：%s",
			resp.StatusCode, strings.TrimSpace(string(body)))}
	}

	var full strings.Builder
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1<<20) // 单行 JSON 可能较长，放大扫描缓冲上限
	for scanner.Scan() {
		var chunk ollamaStreamChunk
		if err := json.Unmarshal(scanner.Bytes(), &chunk); err != nil {
			continue // 忽略空行等无法解析的杂项行
		}
		if chunk.Response != "" {
			full.WriteString(chunk.Response)
			onToken(chunk.Response)
		}
		if chunk.Done {
			break
		}
	}
	if err := scanner.Err(); err != nil {
		return full.String(), err
	}
	return full.String(), nil
}

// ---------------- Prompt 构造 ----------------

// buildPrompt 按固定模板构造提示词：（忌口过滤后的）菜品库 JSON + 用户需求 + 严格的输出格式要求。
// 关键点：
//   - 菜品库序列化为 JSON 后整体拼入，模型据此“查库”而不是凭空编菜名；
//   - 明确要求“不要XX”要对照 ingredients 字段过滤（提示词层保险，
//     硬约束另由后端忌口预过滤兜底保证）；
//   - 强制只返回纯 JSON 数组，便于后端解析。
func buildPrompt(menu []Dish, query string, count int) string {
	menuJSON, _ := json.Marshal(menu) // 序列化菜品库（紧凑 JSON）
	return fmt.Sprintf(`你是一个智能点餐助手。这里有一份菜品库：%s。
用户的需求是："%s"。
请从菜品库中挑选出符合用户要求的菜品。特别注意：如果用户说"不要XX"，一定要检查菜品的 ingredients 中是否包含该成分。
用户需要 %d 道菜。请直接返回一个 JSON 数组，格式为 ["菜名1", "菜名2"]，不要输出任何其他解释文字。`, menuJSON, query, count)
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

// selectDishes 解析模型输出并映射回（忌口过滤后的）可选菜单：
//  1. 提取 JSON 数组；
//  2. 剔除可选菜单中不存在的菜名（模型编造，或已被忌口过滤）；
//  3. 按用户要的数量截断。
func selectDishes(text string, count int, menu []Dish) ([]Dish, error) {
	names, err := extractJSONArray(text)
	if err != nil {
		return nil, err
	}
	log.Printf("模型返回菜名: %q", names)

	// 可选菜单索引：模型返回的菜名必须来自这里（防止编造/绕过忌口过滤）
	allowed := make(map[string]Dish, len(menu))
	for _, d := range menu {
		allowed[d.Name] = d
	}

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
		log.Printf("已剔除不在可选菜单中的菜名: %q", unknownNames)
	}
	if len(selected) == 0 {
		return nil, fmt.Errorf("模型返回的菜名均不在可选菜单中: %v", names)
	}
	if len(selected) > count {
		selected = selected[:count]
	}
	return selected, nil
}

// ---------------- 请求准备（普通/流式接口共用） ----------------

// recommendPlan 一次推荐请求的准备工作产物
type recommendPlan struct {
	Count     int      // 用户要几道菜
	Menu      []Dish   // 忌口过滤后的可选菜品
	Prompt    string   // 构造好的完整提示词
	Forbidden []string // 识别到的忌口关键词
}

// prepareRecommend 解析数量 -> 忌口预过滤 -> 构造 Prompt。
// 忌口过滤后菜品库为空时返回带友好提示的 error。
func prepareRecommend(query string) (*recommendPlan, error) {
	count := parseCount(query)

	// 忌口预过滤（后端兜底）：小模型经常无视提示词里的“不要XX”约束，
	// 因此在把菜品库交给模型之前，先按用户输入确定的忌口食材/口味过滤菜品库，
	// 被禁的菜根本不会出现在模型的可选范围内，保证约束一定生效。
	menu := dishes
	forbidden := parseForbidden(query)
	if len(forbidden) > 0 {
		menu = filterByForbidden(dishes, forbidden)
		log.Printf("识别到忌口: %q，可选菜品从 %d 道过滤为 %d 道", forbidden, len(dishes), len(menu))
	}
	if len(menu) == 0 {
		log.Printf("忌口 %v 过滤后菜品库为空", forbidden)
		return nil, fmt.Errorf("菜品库中没有符合忌口 %v 的菜品，换个忌口试试？", forbidden)
	}

	return &recommendPlan{
		Count:     count,
		Menu:      menu,
		Prompt:    buildPrompt(menu, query, count),
		Forbidden: forbidden,
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

	// 解析数量、忌口过滤、构造 Prompt（流式接口复用同一套准备逻辑）
	plan, err := prepareRecommend(query)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, errorResponse{Error: err.Error()})
		return
	}
	log.Printf("收到需求: %q（需要 %d 道菜）", query, plan.Count)
	// 打印完整模型输入（Prompt），便于调试 Prompt 效果、排查“AI 不听话”的问题
	logPrompt(plan.Prompt)

	// 调用 Ollama。小模型输出偶发不稳定（夹带解释文字、编造菜名），解析失败自动重试一次；
	// 连接类错误（Ollama 没启动等）重试无意义，立即结束。
	var lastErr error
	for attempt := 1; attempt <= 2; attempt++ {
		start := time.Now()
		text, err := callOllama(plan.Prompt)
		if err != nil {
			lastErr = err
			log.Printf("Ollama 调用失败（耗时 %v）: %v", time.Since(start), err)
			break
		}
		// 打印模型原始输出（不截断），便于定位解析失败的具体原因
		log.Printf("========== 模型输出 [耗时 %v] ==========\n%s\n========== 模型输出结束 ==========",
			time.Since(start), text)
		selected, err := selectDishes(text, plan.Count, plan.Menu)
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
	logPrompt(plan.Prompt)

	// 与普通接口相同的重试策略：解析失败自动重试一次。
	// 每次尝试的模型输出都实时推给前端，重试前用 status 事件告知。
	for attempt := 1; attempt <= 2; attempt++ {
		if attempt > 1 {
			writeSSE(w, flusher, "status", map[string]string{"message": "输出解析失败，自动重试中…"})
		}
		start := time.Now()
		text, err := callOllamaStream(r.Context(), plan.Prompt, func(tok string) {
			writeSSE(w, flusher, "token", map[string]string{"text": tok})
		})
		if err != nil {
			// 客户端主动断开（如关闭页面）时静默结束
			if r.Context().Err() != nil {
				log.Printf("客户端断开连接，停止本次流式推荐")
				return
			}
			log.Printf("Ollama 调用失败（耗时 %v）: %v", time.Since(start), err)
			msg := "AI 暂时没想好，请换个说法试试"
			var ollErr *ollamaError
			if errors.As(err, &ollErr) {
				msg = ollErr.msg
			}
			writeSSE(w, flusher, "error", errorResponse{Error: msg})
			return
		}
		// 打印模型原始输出（不截断），便于定位解析失败的具体原因
		log.Printf("========== 模型输出 [耗时 %v] ==========\n%s\n========== 模型输出结束 ==========",
			time.Since(start), text)

		selected, err := selectDishes(text, plan.Count, plan.Menu)
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
