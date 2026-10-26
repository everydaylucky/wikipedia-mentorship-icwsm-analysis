# Wikipedia Mentor Program — 新人提问分类标注手册（Codebook）

## 概述

本手册用于对Wikipedia Mentor Program中新人（mentee）发给mentor的第一条消息进行分类标注。每条消息需要回答5个问题（Q0为前置筛选，Q2-Q5为二元是/否）。Q2-Q5之间不互斥，一条消息可以同时满足多个维度。

理论基础：Q2-Q4基于组织社会化理论中的新人信息寻求类型（Morrison 1993; Miller & Jablin 1991），分别对应Referent（角色定位）、Appraisal（反馈）、Normative（规范）三类信息需求。不同信息需求类型意味着mentor需要采取不同的回复策略，因此对回复有效性的分析需要区分这些类型。

**标注原则**：
- 只看消息文本本身，不看mentor的回复或后续对话
- 不推断用户的隐含动机，只编码文本中可观察的特征
- 遇到模糊case，倾向保守（标否）

---

## 快速参考卡

标注时先看这张表，有疑问再查下方详细定义。

| 代号 | 问题 | 取值 | 一句话判断 | 关键词/信号 |
|------|------|------|-----------|------------|
| **Q0** | 实质性提问/请求/贡献意愿？ | 是/否 | 有没有mentor可以回应的东西？ | 有问号、有"how/what/why"、有"I want to..."→倾向是。纯"Hello"/纯"Thanks"→否 |
| **Q2** | 缺少任务方向？ | 是/否 | 用户是否**不知道做什么**？ | 无方向（"where do I start"）→是。泛泛创建（"create an article/page"无类别）→是。有具体类别（"biography"/"about someone"）→否。有具体topic→否。单步操作（"add image"）→否 |
| **Q3** | 请求反馈？ | 是/否 | 用户是否期望mentor**查看具体内容**并评价？ | "review"、"look at"、"let me know if"、"is this okay"→是。"help me publish"→否 |
| **Q4** | 寻求规范信息？ | 是/否 | mentee 的文本是否**明确表达了对规则/规范/权限的寻求**？ | **主动型**："allowed"/"rules"/"policy"/"guidelines"/"violations"→是。**政策概念名**："notability"/"conflict of interest"/"reliable source"等→是。**事后型**："blocked"/"deleted"/"reverted"/"declined"/"why was my..."→是。仅提到敏感对象但无规范寻求语言→否。纯操作步骤（"how do I add a heading"）→否。⚠ "best practice"/"recommended"等词不自动触发Q4——需看mentee是否在问**规则/标准**，还是在问**实用建议** |
| **Q5** | 已经做了工作？ | 是/否 | mentee 是否**带着已在Wikipedia上产生可观察结果的编辑活动**来提问？ | "my draft"、"my article"、"I edited/created..."、"I have a draft ready"→是。"I want to create..."/"I'm working on..."→否（计划/准备中≠已有工作） |

**标注顺序**：Q0 → 如果Q0=是则标Q2/Q3/Q4/Q5，否则Q2-Q5=N

**快速判断流程**：

```
1. 读完消息，问自己：mentor能回应什么？
   → 什么都回应不了（纯问候/纯感谢/无关内容）→ Q0=否，Q2-Q5=N
   → 有问题/请求/贡献意愿 → Q0=是，继续

2. Q2：用户知不知道自己要做什么？
   → 完全无方向（"where to start"/"what to edit"）→ 是
   → 泛泛的创建任务（"create an article"/"make a page"，无类别限定）→ 是
   → 有具体类别（"biography"/"about an artist"/"add someone"）→ 否
   → 有具体topic（"article about X"）→ 否
   → 单步操作（"add image"/"add citation"/"fix error"）→ 否

3. Q3：用户有没有说"请你看我的东西"并给反馈？
   → 有（review/check/look at + 求反馈）→ 是
   → 只是请求帮忙做某事（publish/move/approve）→ 否

4. Q4：mentee 是否在**寻求规范/规则信息**？
   → 主动型：有政策词（allowed/rules/policy/permitted/guidelines/violations）或权限问句（"is it ok to"/"am I allowed"）→ 是
   → 事后型：工作被政策执行影响（blocked/deleted/reverted/declined/rejected/removed/warned）或追问原因（"why was my..."/"why can't I..."）→ 是
   → 仅提到敏感对象（person/company/self）但无上述规范寻求语言 → 否
   ⚠ 与Q2区分："how to create a page"→ Q2=是, Q4=否；"am I allowed to create a page"→ Q4=是

5. Q5：mentee 是否已经做了工作？（反映社会化阶段）
   → 有证据表明编辑已在Wikipedia上发生（my draft/I edited/was reverted/I created）→ 是
   → 计划、意向或准备中（I want to/I'd like to/I'm working on）→ 否
```

---

## Q0：这条消息是否包含与维基百科编辑相关的实质性提问或请求？

### 定义
用户的消息是否能让 mentor 给出**有实质差异的回复**？如果 mentor 只能回 generic welcome template 或"这不是 Wikipedia 的问题"，则 Q0=否。如果 mentor 可以针对用户的具体情况给出有差异的指导，则 Q0=是。

### 判断标准
核心判据：**mentor 的回复是否会因为这条消息的内容而产生有意义的差异？**

**Q0=是** — mentor 可以给出针对性回复的消息：
- 有关于 Wikipedia 编辑/贡献的问题或请求
- 有关于 Wikipedia 账号/流程/社区的具体问题（包括 username、password recovery、dark mode 等——只要和 Wikipedia 平台相关）
- 表达了明确的编辑/贡献意愿（即使没有具体问题）

**Q0=否** — mentor 只能给出 generic 回复的消息，完整列表：
1. **纯问候**："Hello!" / "Hi!" / "Nice to meet you!"（无任何后续内容）
2. **纯感谢**："Thanks!" / "Thank you for the welcome!"
3. **太短/无意义**：单个词、乱码、无法理解的内容
4. **非 Wikipedia 相关**：Facebook/Gmail/Instagram/WiFi 密码、手机问题、Google 搜索问题、求职、Wikipedia 是否付钱等；也包括请求 Wikipedia 不承载的内容类型（Wikipedia 是百科全书，误解其性质的请求归此类）
5. **非英语**：消息主体不是英语
6. **问是不是 bot**："Are you a real person?" / "Are you a bot?"（无其他实质内容）
7. **分享与维基百科编辑无关的经历/感受且无具体问题**："It is my first day to join Wikipedia and I am so excited!" / "I have no questions for now"（纯分享，无编辑/贡献意愿，无请求。注意：如果提到了自己的编辑行为或贡献意愿，如 "Put in my first edit today!"，则属于 Q0=是，因为表达了明确的编辑/贡献意愿）
8. **纯自我介绍无请求**："hello I am a 10 y/o kid" / "I am Paul"


⚠ 注意：以下看似"非编辑"但实际 Q0=是：
- **Username 问题**：涉及 attribution policy，mentor 给实质性指导 → Q0=是
- **Password/账号恢复**：涉及账号政策和流程，mentor 给具体建议 → Q0=是
- **Dark mode/界面设置**：mentor 教具体操作步骤（gadgets/CSS）→ Q0=是
- **User page 编辑**：属于 Wikipedia 社区参与活动 → Q0=是

### 是的例子

| 消息 | Q0 | 说明 |
|------|-----|------|
| "How do I add a citation to an article?" | 是 | 操作性问题 |
| "My draft was declined, what happened?" | 是 | 有具体问题，涉及规则 |
| "Can you look at my article?" | 是 | 有具体请求（review） |
| "I want to start writing articles, how do I begin?" | 是 | 有具体问题（怎么开始） |
| "How do I change my signature?" | 是 | Wikipedia 平台操作 |
| "How do I change my username?" | 是 | 涉及 attribution policy，mentor 给实质性指导 |
| "How do I activate dark mode?" | 是 | mentor 教 gadgets/CSS 具体操作 |
| "I forgot my password, how to recover my account?" | 是 | 涉及 Wikipedia 账号政策和恢复流程 |
| "Fix my userpage" | 是 | User page 编辑=Wikipedia 社区活动 |
| "I'm new here and excited to contribute!" (无具体问题) | 是 | 表达了明确的贡献意愿=隐含引导请求 |
| "Can a Wikipedia page be created about me without my consent?" | 是 | 条目主体问规则（consent=权限问题） |
| "How do I get my page removed?" | 是 | 条目主体请求修改 Wikipedia 内容，有具体页面 |
| "Someone wrote false information about me, how do I get it taken down?" | 是 | 条目主体请求修正信息，属于 Wikipedia 内容相关 |
| "Put in my first edit today! Excited to venture forth" | 是 | 提到了自己的编辑行为→有明确贡献意愿 |

### 否的例子

Q0=否时，Q2-Q5=N。

| 消息 | Q0 | 说明 |
|------|-----|------|
| "Hello! Nice to meet you!" | 否 | ①纯问候 |
| "Thank you for the welcome message!" | 否 | ②纯感谢 |
| "No" / "W" / "Earthh" | 否 | ③太短/无意义 |
| "What is energy?" | 否 | ④非 Wikipedia 相关（知识问题） |
| "Facebook locked" / "Instagram password cracking" | 否 | ④非 Wikipedia 相关（其他平台） |
| "How can I get the Wi-Fi password" | 否 | ④非 Wikipedia 相关 |
| "Does Wikipedia pay editors?" / "how do I start earning" | 否 | ④非 Wikipedia 相关（Wikipedia 不付费） |
| "I'm trying to put my information on Google" | 否 | ④非 Wikipedia 相关（Google 问题） |
| "Are you a real person?" (无其他内容) | 否 | ⑥问是不是 bot |
| "It is my first day to join Wikipedia and I am so excited" | 否 | ⑦分享感受，无编辑/贡献意愿，无具体问题 |
| "hello I am a 10 y/o kid" (无后续问题) | 否 | ⑧纯自我介绍无请求 |
| "free XVideos red" | 否 | ③无意义内容 |

### 边界案例

| 消息 | Q0 | 说明 |
|------|-----|------|
| "Hello, I created an account only because I thought that I was required to do so in order to use the site. Am I correct?" | 否 | 问的是账号是否必须，不涉及编辑活动 |
| "I don't understand Wikipedia" | 否 | 太模糊，没有具体可回应的请求 |
| "I'm new here and excited to contribute! But this UI is really confusing, I don't know where to start" | 是 | "不知道从哪开始"=有实质请求 |
| "I am makw same mistakw in this web sob forgive me i will not occur this in future" | 否 | 道歉，没有提问或请求 |

---

## Q2 Referent：用户是否缺少任务方向——不知道做什么、从哪开始？（是=缺方向，否=有方向）

### 定义
用户**缺少 task clarity**——不知道该做什么、该把精力放在哪里、该从哪里入手。这对应 Morrison 1993 的 Referent information seeking：新人在寻求关于角色和任务要求的信息。

### 判断流程

```
1. 用户是否完全没有方向？
   → "how to start editing" / "where to begin" / "what should I edit" → Q2=是
   → 有一个想做的任务 → 继续第2步

2. 任务描述是否有具体类别或 topic？
   → 有具体 topic（"article about Odin's Valkyries"）→ Q2=否
   → 有具体类别（"biography" / "about an artist"）→ Q2=否（带着具体目的来的）
   → 只用了泛化词（article / page / wiki）→ 继续第3步

3. 泛化任务 vs 单步操作？
   → create/write/make + article/page/wiki → Q2=是（太笼统，等同于缺方向）
   → add image / add citation / fix error 等 → Q2=否（单步操作）
```

### 复杂任务清单 vs 单步操作清单

标注员不需要自行判断任务复杂度，直接查表：

**泛泛的创建任务（Q2=是）** — 这些表述太笼统，用户没有表达任何具体方向，和"不知道从哪开始"本质上无法区分：
- create/write/make/start/add + **article** / **page** / **wiki** / **entry** / **new page** / **new article**
- 关键特征：只用了最泛化的动词+最泛化的对象（article, page, wiki），没有任何类别限定
- ⚠ 注意：**post/publish/submit + article/page** 不属于此类 → Q2=否。这些动词暗示用户已有内容，问的是发布流程，属于单步操作

**有具体类别的创建任务（Q2=否）** — 一旦用户指定了比 "article/page/wiki" 更具体的内容类型或对象类别，说明他们带着具体目的来，不缺方向。判断规则：**比 article/page 更具体一层就算有方向。** 常见类别包括：
- **人物类**：biography / profile / someone / a person / "add someone to Wikipedia"
- **组织类**：company / organization / business / brand / agency / church
- **影视/文化类**：film / movie / TV show / series / album / song / band / book / novel
- **地点类**：place / city / town / village / country / school / university
- **体育类**：team / player / match / athlete
- **事件类**：event / incident / battle / war
- **Wikipedia 特定类型**：template / userbox / user page / list / redirect / disambiguation / category page / draft / essay / portal / stub / infobox / help page / talk page / WikiProject page / file page / media page
- **领域限定**：write about **a topic I know** / contribute on **AI/technology** / edit **history articles**

**单步操作（Q2=否）** — 这些任务是自足的操作指令，mentor 可以直接教步骤：
- **post** / **publish** / **submit** + article / page / draft（已有内容，问发布流程）
- add/insert/upload **image** / **picture** / **photo**
- add/insert **citation** / **reference** / **source**
- add/insert **infobox** / **template** / **userbox**
- add/insert **link** / **URL** / **external link** / **wikilink**
- add/insert **table** / **chart**
- **edit** a section / a paragraph / **fix** an error / **correct** a mistake
- **format** text / add **bold** / add **italic** / add **heading** / add **subheading**
- **change** signature / change username
- **move** a page / **rename** a page / **redirect** a page
- add **categories** / add **tags**
- **revert** an edit / **undo** an edit

### 判断标准总结
- **完全无方向**（"where to start"/"what to edit"）→ Q2=是
- **泛泛的创建任务**（create/write + article/page/wiki，无类别限定）→ Q2=是
- **有具体类别**（biography/profile/company page/about an artist）→ Q2=否
- **有具体 topic**（"article about Odin's Valkyries"）→ Q2=否
- **单步操作**（add image/citation/infobox 等）→ Q2=否
- **已经在编辑中遇到具体障碍** → Q2=否
- **"don't know where to start"**：如果有具体 topic，按 Q2=否处理（"不知从哪开始"=问流程）；如果无具体 topic，按 Q2=是处理
- 如果"any advice"/"any suggestions"是消息的**核心请求**→Q2=是；如果是描述完具体问题后的**礼貌收尾**→Q2=否

### 是的例子

| 消息 | Q2 | 说明 |
|------|-----|------|
| "How can I start my editing on Wikipedia?" | 是 | 经典Referent：不知道从哪开始 |
| "What should I edit? I know a lot about computers but I don't know what to edit" | 是 | 有能力但缺方向，最纯的Referent |
| "Which kind of pages should I edit and which kind of pages should I avoid?" | 是 | 寻求优先级和边界 |
| "where do you start when it comes to editing? is there a recommended place to start?" | 是 | 明确求方向 |
| "I would like to translate articles from English to my language. But I don't know where to start" | 是 | 有目标但缺路径 |
| "Can you mentor me through this whole process. I would really appreciate." | 是 | 求全程引导 |
| "Hello, teach me how to edit" | 是 | "teach me"暗示系统性学习需求 |

### 否的例子

| 消息 | Q2 | 说明 |
|------|-----|------|
| "I am new to Wikipedia, but have long time interest in Caucasus regional politics. I want to write about the regional politics and started with one of the politician — I am not sure if I should proceed with other pages or just wait for approval" | 否 | 有具体项目但在问策略（继续还是等待） |
| "How do I create an article about Odin's Valkyries?" | 否 | 有具体topic→Q2=否 |
| "I want to create a biography" | 否 | "biography"=具体类别→Q2=否 |
| "Hello, please how do you add someone to wikipedia?" | 否 | "add someone"=创建人物页面，有具体类别（人物）→Q2=否 |
| "How do I add an infobox?" | 否 | 单步操作清单→Q2=否 |
| "How do I add a citation?" | 否 | 单步操作清单→Q2=否 |
| "How do I add a link to the references section?" | 否 | 单步操作清单→Q2=否 |
| "How do I change my signature?" | 否 | 单步操作清单→Q2=否 |
| "How to i post a wiki" | 否 | "post"=发布操作，暗示已有内容→Q2=否 |
| "I'm interested in history, any suggestions for what to work on?" | 否 | 有兴趣领域（history=具体类别）→Q2=否。"any suggestions"是在领域内找具体 topic，不是缺方向 |
| "Can you review my draft?" | 否 | 有具体工作，求的是评价不是方向 |
| "Why was my edit reverted?" | 否 | 不是在找方向，是在问规则 |
| "Please move my page to mainspace" | 否 | 请求执行，不是求方向 |
| "is it recommended to use less citations or more citations for short articles?" | 否 | 问社区惯例/最佳实践，不是缺方向 |

### 边界案例

| 消息 | Q2 | 说明 |
|------|-----|------|
| "Hello am I wanna know how to edit my Wikipedia" | 是 | 无具体目标/topic→Q2=是 |
| "Any advice you have re editing or creating Wikipedia entries most welcome" | 是 | 泛泛的"any advice"作为消息主要请求→Q2=是 |
| "how to create an article about Odin's Valkyries but I do not know where to start" | 否 | 虽然说了"don't know where to start"，但有具体 topic（Odin's Valkyries）→ Q2=否。"不知从哪开始"在有具体 topic 时=问流程，不是缺方向 |
| "I went to the page for Joel Hurt and added his grandchild...I'm open to any advice" | 否 | "any advice"跟在具体问题描述后面，是礼貌收尾。用户有明确目标 |
| "What can I do about this?" (编辑被人反复回退) | 否 | "what can I do"指向具体困境的解决方案，不是缺方向 |

---

## Q3 Appraisal：用户是否在请求对自己已有工作的反馈或评估？

### 定义
用户已经做了某项工作（写了草稿、做了编辑、建了页面），现在请求别人**看自己的工作并给出判断**——做得怎么样、哪里要改、是否足够好。

### 判断标准
核心判据：**用户是否期望mentor去查看具体内容并给出评价/反馈/改进建议？**
- "review"、"check（并给反馈）"、"look at...let me know"、"is this okay"、"how can I improve"→Q3=是
- "help me publish"、"approve my draft"、"when will it be approved"→Q3=否（请求的是动作或信息，不是评价）
- ⚠ **针对具体作品的发布/提交指导**：当用户针对一篇具体的已有文章寻求发布/提交方面的**指导或建议**（advise, best way to proceed, how should I）时，需要判断：mentor 是否必须打开并阅读用户的内容才能给出有意义的回答？如果是 → Q3=是（隐性求评价）；如果 mentor 只需给出通用流程步骤即可回答 → Q3=否。例如："Please advise on the best way to proceed with publishing this article"（针对具体文章求指导）→ Q3=是；"How do I publish an article?"（泛泛问流程）→ Q3=否

### 是的例子

| 消息 | Q3 | 说明 |
|------|-----|------|
| "Can you review my article in draft and let me know if it will be moved to the main space?" | 是 | 明确请求review |
| "Can you take a look at the article I created and let me know if there is anything that I need to fix?" | 是 | 求反馈+改进建议 |
| "I have formulas and figures in my drafts. Are there problems?" | 是 | 求作品检查 |
| "Please can you help check this article out and let me know how I can improve it? I've been working on it for a while now." | 是 | 求改进建议 |
| "I've been going a bit ham on editing Wikipedia, and there are a few pages where I made some pretty hefty edits. If you have some time and would be willing to take a look...it would be great to know whether I'm headed in the right direction" | 是 | 求方向确认 |
| "I've created a draft that I would like to publish. Would you be able to review it?" | 是 | "review"=看了给反馈 |
| "I created a page. Please take a look and let me know if everything is done the right way" | 是 | 求确认 |
| "this is my first wikipedia page...in the sandbox. Please let me know if this is fine so far" | 是 | 求反馈 |
| "for my article I get the feeling that more citations are better but not sure if that is the case: [LINK]" | 是 | 带着具体作品问最佳实践 |
| "I look forward to your critique and suggestions on how I can improve" | 是 | 明确求critique |
| "I'd really like some advice on my contribution to the Mad Max: Fury Road page" | 是 | 求对自己贡献的反馈 |
| "I've been working on it for 2 months but didn't get response, only response I got is declined. I don't know what to change on that article, can you help me out with this?" | 是 | "不知道改什么"=需要有人看了告诉我 |
| "Can you kindly review my paid editing disclosure and confirm it's compliant?" | 是 | 求合规审核 |
| "Can you assist to verify the content before publish?" | 是 | 求审核 |
| "I posted my first page on Sunday...Can you please review the posting and kindly let me know what needs to be done to have it accepted?" | 是 | 明确请求review+求改进方向 |
| "I've enjoyed completing the tasks...When and how should I decide whether to remove the issues tags? See for example Alan Bircher: [LINK] — I added more thorough documentation..." | 是 | "See for example"+链接+描述改动=隐含请求mentor去看并判断 |

### 否的例子

| 消息 | Q3 | 说明 |
|------|-----|------|
| "I've Drafted an article! Can you help me Publish it?" | 否 | 请求帮忙执行发布，没有任何求反馈的意思 |
| "When will I be able to submit my draft for review?" | 否 | 问流程/时间 |
| "How do I add images to my draft?" | 否 | 操作问题 |
| "Where can I turn my draft to a public article?" | 否 | 问流程 |
| "how long does it take a page to get published?" | 否 | 问流程 |
| "How do I publish an article?" | 否 | 操作问题 |
| "could you advise when my article will be approved for publishing?" | 否 | 问时间，不是求评价 |
| "Please approve My draft" | 否 | 用户以为mentor有审批权限，在请求执行动作 |
| "How do I know if the article I edited has improved?" | 否 | 问的是评估方法/标准，不是请求别人来评价。如果说"has my article improved?"则Q3=是 |
| "Any chance you're willing to make some edits so this is strong enough to submit?" | 否 | 请求mentor动手改（Service），不是求反馈 |
| "on my edits there are numbers like (+118)...what do those mean?" | 否 | 问界面功能 |
| "I got warned cos my editing was shyt" | 否 | 报告事件，没有请求反馈 |
| "Any advice you have re editing or creating Wikipedia entries most welcome" | 否 | 泛泛求建议，不是对具体作品求反馈。Q2更合适 |

### 边界案例

| 消息 | Q3 | 说明 |
|------|-----|------|
| "am I doing well? 44 edits not bad" | 是 | 求整体表现评价，虽然没指向具体作品 |
| "you noted that my contribution lacked notability. I disagree..." | 否 | 在回应已有appraisal并寻求重新评估，更像质疑/申诉 |
| "I created an account to edit an Article and create one about Marcel Bonin. I know that Wikipedia needs sources and I wondered how all that works" | 否 | 虽然提到了自己的项目，但问的是操作方法（怎么加来源），不是求评价 |
| "The article is quite ready to be published...I need to do at least 10 reviews to publish. Am I right?" | 否 | 问流程规则，不是求对文章的评价 |

---

## Q4 Normative：mentee 是否在寻求规范/规则信息？

### 定义
mentee 的消息文本是否**明确表达了对规则、规范、权限信息的寻求**？Q4 编码的是 mentee 的信息寻求行为（Morrison 1993 Normative information seeking），不是推测 mentor 的回复是否会涉及政策。

Q4 的覆盖范围与 Morrison 1993 的 normative information 定义一致——包括组织中被期望的行为方式的各个层面，不限于强制执行的正式政策。Wikipedia 的规范体系有四个层级，Q4 覆盖全部：
- **Policies**（notability, BLP, COI）→ 强制执行
- **Guidelines**（reliable sources, NPOV）→ 有执行力
- **Manual of Style**（heading 格式、标点规范）→ 社区共识的格式规范
- **Essays/惯例**（段落长度、citation 数量偏好）→ 社区最佳实践

**核心原则**：只看 mentee 说了什么，不看 mentee 提到的对象是否客观上涉及政策。即使 mentee 提到了 living person / company / self 等敏感对象，只要 mentee 的语言中没有规范寻求信号，就标 Q4=否。

### 判断标准
两类文本信号（至少出现一类 → Q4=是，都没有 → Q4=否）：

**类型 A：主动规范寻求（Proactive Normative）** — mentee 在行动前主动询问规则/权限/合规性：
- **规范寻求语言**（每个词配典型语境）：
  - **allowed/permitted**："Am I allowed to edit this page?" / "Is it permitted to use YouTube as a source?"
  - **rules/policy/guidelines/terms**："What are the rules for creating a new page?" / "I want to follow the correct guidelines"
  - **violations/violating**："I don't want to violate any Wikipedia terms" / "Is this a violation?"
  - **consent/authorize**："Can a page be created without my consent?" / "I didn't authorize this page about me"
  - **permission**："Do I need permission to edit a protected page?" / "Who gives permission to publish?"
  - **prohibited/banned/blocked**："Is it prohibited to edit my own company's page?" / "Will I get banned for this?" / "Why am I blocked?"
  - **appropriate/legitimate**："Is it appropriate to add my own research?" / "Is this a legitimate source?"
  - **ok/okay**（合规语境下）："Is it ok to use images from Google?" / "Is it okay if I edit a page about a living person?"
  - **qualify/eligible/criteria**："Does this topic qualify for a Wikipedia page?" / "Am I eligible to create articles?" / "What are the criteria for notability?"
  - **compliant/compliance**："Is my disclosure compliant?" / "I want to make sure I'm in compliance with Wikipedia's policies"
  - **legal/illegal**："Is it legal to copy text from another website?" / "Would it be illegal to use this image?"
  - **proper/properly**（合规语境下）："How to properly cite sources without getting in trouble?" / "What is the proper way to disclose paid editing?"（注意区分：如果 properly 只是指操作正确性如 "how to properly format a table" → Q4=否）
- **隐性合规询问（自举证模式）**：mentee 没有直接用政策词，但**主动列举了自己内容可能不符合标准的理由**，然后问"能不能做/能不能发布"。这种"先自举证潜在问题 → 再问是否可行"的结构说明 mentee 意识到了合规风险，本质上是在问"这种内容是否被 Wikipedia 接受"。判断要点：
  - ✅ Q4=是："there is no official proof" / "it is completely fantasy" / "only written on paper" + "will this be possible to publish?" — mentee 主动提供了可能违反 notability/verifiability 的因素，问的不是"怎么发布"（操作），而是"这样的内容能不能发布"（合规性）
  - ✅ Q4=是："he's not very well known and there aren't many sources about him, can I still make a page?" — 主动说明来源不足，问是否可行
  - ❌ Q4=否："I want to create a page about my fantasy wrestling league, how do I start?" — 没有列举任何潜在问题，纯操作问题
  - **区分关键**：单纯问"can I do X"→ 操作问题→Q4=否；问"can I do X"并且**主动说明了 X 可能不符合标准的原因**→ 合规询问→Q4=是
- **权限门槛/准入要求**："how many edits to get privileges" / "when can I create my own pages" / "confirmed user" / "how many/much X do I need to pass/be allowed/get accepted"（句式为"需要多少X才能被允许/通过Y"时，说明mentee在询问某种标准或门槛，属于normative seeking）
- **许可范围/边界询问**："what type of X can I put/add/share" / "which type of links are allowed"——当 mentee 问"什么类型的内容/链接/信息可以放"时，本质上是在询问什么是被许可的、什么是不被许可的，这是对规范边界的探询。区分：如果问的是"怎么加链接"（how to add a link）→ 纯操作→Q4=否；如果问的是"什么样的链接可以加"（what type of links to add）→ 问许可范围→Q4=是
- **政策概念名** — mentee 在**询问或讨论**某个 Wikipedia 政策概念的规则/标准/适用性时 → Q4=是。注意：仅仅提到政策概念名作为已知词汇来描述其他事情不算。例如："What counts as a reliable source?" → 问规则 → Q4=是；"I found a reliable source and want to add it" → 描述性使用 → Q4=否。"What counts as vandalism?" → 问规则 → Q4=是；"What pages do you find most frequently vandalized?" → 用 vandalism 描述方向 → Q4=否。常见政策概念名：
  - 内容标准类："notability" / "notable enough" / "reliable source" / "verifiability" / "neutral point of view" / "NPOV" / "original research"
  - 利益相关类："conflict of interest" / "COI" / "paid editing" / "disclosure"
  - 版权类："copyright" / "fair use" / "public domain" / "free license"
  - 行为规范类："edit war" / "three-revert rule" / "3RR" / "vandalism" / "sock puppet" / "canvassing"
  - 流程类："speedy deletion" / "AfD" / "articles for deletion" / "semi-protected" / "autoconfirmed"

**类型 B：事后规范寻求（Reactive Normative）** — mentee 的工作被政策执行影响，寻求理解原因：
- **执行结果词**："blocked" / "deleted" / "reverted" / "declined" / "rejected" / "removed" / "warned" / "undone" / "locked" / "semi-protected" / "not published" / "not incorporated" / "not accepted" / "disappeared"
- **以执行结果为中心的请求** → Q4=是，包含以下所有情况：
  - (a) 明确追问原因："why was my..." / "why can't I..." / "why did they..." / "what happened to my..."
  - (b) 询问如何解决或撤销："how do I fix this" / "how can I get it back"
  - (c) 表达困惑："what happened?" / "what shall I do?" / "what am I doing wrong?"
  - (d) 仅报告执行结果而没有其他请求——隐含问题仍然是"为什么/我该怎么办"
  - （解决任何政策执行结果都必然需要 mentor 解释触发该结果的底层规则或规范。）
  - 注意：此处 Q4=是 与 Q3 无关。一条消息可以**同时**寻求对工作的反馈（Q3=是）**并且**涉及规范问题（Q4=是）。
- **质疑/申诉**："I would like to contest..." / "I disagree with..." / "how is this fair..."
- ⚠ **陈述背景 vs. 以执行结果为中心**：执行结果词出现时，还要看 mentee 的**主要请求是否围绕执行结果本身**，还是**转向了完全不同的请求**。例如："My first article was deleted (I was not done)...Can you look at my new article?" — mentee 提到了 deleted，但转向了新的请求（review）→ 对转向后的请求单独判断 Q4。对比："My article was deleted, why?" → Q4=是

**Q4=否**：mentee 的文本中没有上述任何规范寻求信号。以下情况即使客观上会触发政策回复，仍标否：
- 仅提到敏感对象（"create a page about a politician" / "write about my company"）但没有规范寻求语言，也没有主动列举可能违规的因素
- 纯操作问句（"how do I create a page about X"）
- "Can I" + 无政策词、无合规语境、无自举证 → 当作操作问题
- ⚠ **提到流程/政策名称 ≠ 寻求规范信息**：mentee 提到了 Wikipedia 的流程或政策概念名，但实际问的是**操作步骤**而非**规则本身**，仍标 Q4=否。关键区分：mentee 是在问"怎么操作"还是在问"规则是什么/是否被允许"。例如：
  - "How do I submit through Articles for Creation?" → 问操作步骤（找不到按钮在哪）→ Q4=否
  - "What are the criteria for Articles for Creation to accept my draft?" → 问审核标准/规则 → Q4=是
  - "How do I submit for speedy deletion?" → 问操作 → Q4=否
  - "What qualifies for speedy deletion?" → 问规则 → Q4=是
- ⚠ **容易与规范混淆的操作性关键词**（均为 Q4=否）：
  - "what formats are accepted" → 技术文件格式规格，Q4=否
  - "how do I create/make/start" → 操作步骤，Q4=否
  - "can I delete/undo/revert" → 操作能力问题，Q4=否
  - "how many articles can we write" → 事实性平台能力问题，Q4=否
  - "what do I have to do before X is possible" → 流程步骤，Q4=否
- ⚠ **"best practice"/"recommended"/"should I" 不自动触发 Q4**：这些词常出现在寻求实用建议的语境中，mentor 的回答是经验性的操作建议而非引用政策。例如：
  - "is it recommended to use less citations or more?" → 问实用建议，Q4=否
  - "how many citations should I add?" → 问实用建议，Q4=否（没有 Wikipedia 政策规定 citation 数量）
  - "what is the best practice for sourcing?" → 问实用建议，Q4=否
  - 对比："is it recommended **per Wikipedia policy** to use less citations?" → 明确引用政策 → Q4=是

### 与Q2的区分
Q2（路径规划）和Q4（规范寻求）可以同时为是。Q2看方向感，Q4看规范寻求语言：

| 消息 | Q2 | Q4 | 判断依据 |
|------|----|----|---------|
| "How to create a page?" | 是 | 否 | 泛泛创建→Q2=是。无规范寻求语言→Q4=否 |
| "How do I create an article about X?" | 否 | 否 | 有topic→Q2=否。无规范寻求语言→Q4=否 |
| "How do I create a page about a politician?" | 否 | 否 | 有具体类别→Q2=否。无规范寻求语言→Q4=否 |
| "create a page for my company" | 否 | 否 | 有具体类别→Q2=否。无规范寻求语言→Q4=否 |
| "Am I **allowed** to create a page about a politician?" | 否 | 是 | "allowed"=规范寻求→Q4=是 |
| "What type of information can I share on my user page?" | 否 | 是 | "what type of X can I share"=问许可范围→Q4=是 |
| "How do I add a link to my user page?" | 否 | 否 | "how do I add"=纯操作问题→Q4=否 |
| "Which type of links can I add on my user page?" | 否 | 是 | "which type of X can I add"=问什么类型被允许→Q4=是 |
| "I want to create my own biography, is that **against the rules**?" | 否 | 是 | "against the rules"=规范寻求→Q4=是 |
| "How to add an image?" | 否 | 否 | 单步操作→Q2=否。无规范寻求语言→Q4=否 |
| "Am I **allowed** to add an image?" | 否 | 是 | "allowed"=规范寻求→Q4=是 |
| "My edit was **reverted**, why?" | 否 | 是 | "reverted"+"why"=事后规范寻求→Q4=是 |

### "Can I do X" 的区分规则
"Can I" 在英语中既可以表示"技术上能不能"也可以表示"规则上允不允许"。判断依据：
- "Can I" + **政策词/合规语境**（"allowed", "permitted", "without getting in trouble", "conflict of interest", "copyright"等）→ Q4=是
- "Can I" + **无政策词、无合规语境** → Q4=否，当作操作问题
- ⚠ 不根据对象是否敏感来推断 "Can I" 的含义——mentee 必须自己用语言表达合规关切

**"Can I" Q4=否的典型例子**（纯操作/技术能力/平台功能）：

| 消息 | Q4 | 说明 |
|------|-----|------|
| "Can I edit a page on my phone?" | 否 | 问技术能力 |
| "Can I add bold text to an article?" | 否 | 问格式操作 |
| "Can I undo my last edit?" | 否 | 问操作步骤 |
| "Can I save a draft and come back later?" | 否 | 问平台功能 |
| "Can I see who edited a page?" | 否 | 问平台功能（编辑历史） |
| "Can I edit a page without creating an account?" | 否 | 问平台能力（匿名编辑） |
| "Can I translate an article from another language?" | 否 | 问操作流程 |
| "Can I create a page about my professor?" | 否 | 仅对象敏感，无政策词 |
| "Can I write about my company?" | 否 | 同上 |
| "Can I make my own biography?" | 否 | 同上 |

**同一对象，Q4=否 vs Q4=是的对比**：

| Q4=否（无政策词） | Q4=是（有政策词） | 区别 |
|-------------------|-------------------|------|
| "Can I create a page about my company?" | "Can I create a page about my company without it being a **conflict of interest**?" | 加了政策概念名 |
| "Can I write about myself?" | "Is it **allowed** to write about myself?" | 加了政策词 |
| "Can I add a photo I found online?" | "Can I add a photo I found online, or is that **copyright** infringement?" | 加了政策概念名 |
| "Can I edit the page about my company?" | "Can I edit the page about my company, or would that **violate** some policy?" | 加了政策词 |
| "Can I create two accounts?" | "Can I create two accounts, or is that **against the rules**?" | 加了政策词 |
| "Can I add my own research findings?" | "Can I add my own research, or does that count as **original research**?" | 加了政策概念名 |

### 是的例子——类型A：主动规范寻求（Proactive Normative）

mentee 在行动前主动询问规则/权限/合规性。

| 消息 | Q4 | 说明 |
|------|-----|------|
| "Am I allowed to put in my own opinions?" | 是 | "allowed"=直接问是否被允许 |
| "I would like to understand why I cannot upload an image of a public figure I found on the internet. Is there a way to use this image under the fair use policy?" | 是 | "fair use policy"=明确提到政策 |
| "I'm still quite confused on the uploading an image rules" | 是 | "rules"=明确政策词 |
| "Can I use YouTube links as a reference?" | 是 | "as a reference"=问什么来源类型被接受→许可范围询问（类似"what type of sources can I use"） |
| "How can I know when I'm a confirmed / auto-confirmed user to start my own pages?" | 是 | 问权限门槛 |
| "how many edits does it take to have editing privileges?" | 是 | "privileges"=权限规则 |
| "I'm being accused of 'gaming' the system. Thoughts?" | 是 | 问行为规范 |
| "I think I broke the three edit war rule" | 是 | "rule"=政策词 |
| "how to properly edit a Wikipedia article without any hassle and avoid any potential violations" | 是 | "violations"=明确政策词 |
| "is it allowed to re-do an entire article given to you for editing for its betterment?" | 是 | "allowed"=直接问是否被允许 |
| "if I find a subject without a page, can I take the liberty and create a quality page?" | 是 | "take the liberty"=不确定是否被允许→合规意识 |
| "How to reliably edit so my edits won't be dismissed as 'Lies'?" | 是 | "dismissed"=担心被政策执行否定 |
| "I would like to follow the correct guidelines so I don't violate any Wikipedia terms" | 是 | "guidelines"+"violate"+"terms"=多个政策词 |
| "I wanted to create a wiki page for the musician 'she hacked you'...How best do I approach this so that I follow the rules?" | 是 | "follow the rules"=明确规范寻求 |
| "I want to create a biography article of my mentor...while staying within the rules of providing independent, third-party references" | 是 | "within the rules"+"independent, third-party references"=明确规范寻求 |
| "Am I allowed to create a page about a politician?" | 是 | "allowed"=政策词→Q4=是（注意：如果没有 "allowed" 只说 "How do I create a page about a politician" → Q4=否） |
| "What are the notability requirements for a Wikipedia article?" | 是 | "notability"=政策概念名→Q4=是 |
| "Do I have a conflict of interest if I edit my company's page?" | 是 | "conflict of interest"=政策概念名→Q4=是 |
| "What counts as a reliable source?" | 是 | "reliable source"=政策概念名→Q4=是 |
| "Is this copyright infringement if I use an image from Google?" | 是 | "copyright infringement"=政策概念名→Q4=是 |
| "I submitted my first draft and was declined because I cited incorrectly...the article has notability due to press coverage" | 是 | "declined"=执行结果 + "notability"=政策概念名→Q4=是 |
| "I want to properly disclose any conflict of interest to make sure everything is transparent" | 是 | "conflict of interest"+"disclose"=政策概念名+合规意识→Q4=是 |

### 是的例子——类型B：事后规范寻求（Reactive Normative）

mentee 的工作被政策执行影响（删除/回退/拒绝/封禁等），寻求理解原因。

| 消息 | Q4 | 说明 |
|------|-----|------|
| "why wikipedia removing my editing lines" | 是 | "removing"=执行结果+"why"=追问原因 |
| "why are people downvoting my edits?" | 是 | "downvoting"=执行结果+"why"=追问 |
| "it is rejected by Wikipedia because I personally didn't write the ad back in 1927...How are these images allowable while mine is not?" | 是 | "rejected"+"allowable"=质疑规则一致性 |
| "I edited the fordite wiki and the edit was deleted...The current wiki contains several false and misleading statements" | 是 | "deleted"=执行结果 |
| "one of my pages has been deleted. I would like to know what was wrong with it" | 是 | "deleted"+"what was wrong"=追问原因 |
| "I would like to contest the speedy deletion of the page because I feel that it was wrongfully deleted" | 是 | "deletion"+"contest"=质疑/申诉 |
| "rejected with the explanation that it constituted advertising...I invested considerable time ensuring the content adhered to Wikipedia's guidelines" | 是 | "rejected"+"guidelines"=事后规范寻求 |
| "All the contributions I have made were reverted. I have not changed the text; I just added a citation..." | 是 | "reverted"=执行结果，隐含"为什么" |
| "My athlete page Barry Nobles was deleted and I can't seem to find out why" | 是 | "deleted"+"why"=追问原因 |
| "Can you help me submit this correctly? I'm planning to submit a new version now after the first version was declined" | 是 | "declined"=执行结果 |
| "Why cannot I add some of the links in the references?" | 是 | "cannot"=被阻止，事后规范寻求 |
| "two of my articles were declined of lack of references" | 是 | "declined"=执行结果 |

### 否的例子——操作问题（无规范寻求语言）

| 消息 | Q4 | 说明 |
|------|-----|------|
| "How Can I publish my article?" | 否 | 操作问题 |
| "How can I edit my page?" | 否 | 操作问题 |
| "How can I add an article?" | 否 | 无规范寻求语言→Q4=否 |
| "Any tips on putting links in the references section?" | 否 | mentor 回复可能涉及 reliable sources 政策，但 mentee 没有在问规则→Q4=否 |
| "How can I summarize an article?" | 否 | 操作问题 |
| "How can I fix it?" (citation error) | 否 | 技术问题 |
| "How do I get started?" | 否 | 方向问题（Q2），不是规则问题 |

### 否的例子——提到敏感对象但无规范寻求语言

⚠ 这些消息的编辑对象客观上可能触及政策（BLP/COI/notability），但 mentee 的文本中没有规范寻求信号，所以 Q4=否。

| 消息 | Q4 | 说明 |
|------|-----|------|
| "How Can I add a page about a politician who is currently active?" | 否 | 虽然 politician=敏感对象，但 mentee 只在问操作→Q4=否 |
| "I need to add an article about my late husband. He was a civil rights leader. How do I get this done?" | 否 | mentee 只在问流程，没有任何规范寻求语言。mentor 会主动提 notability，但那是 mentor 的行为 |
| "how can I create a page for my charity" | 否 | mentee 只在问操作。mentor 会提 COI，但 mentee 没问 |
| "create a page for my company" | 否 | 同上 |
| "I have discovered an upcoming artist and I wanna write about him" | 否 | mentee 在说目标，没有规范寻求语言 |
| "can I make my professor's page in Wikipedia?" | 否 | "can I" + 无政策词 → 当作操作问题→Q4=否 |
| "How do I find my comments on the Andrew Tate page?" | 否 | 操作问题。Andrew Tate 是在世人物但 mentee 没有在问规则 |
| "How can I create my own biography?" | 否 | "can I" + 无政策词 → 操作问题。mentor 会主动提 COI/autobiography policy，但 mentee 没问 |
| "I was reading an article about a company I was working in and saw the logo is their old one. How to change it?" | 否 | 操作问题（换图片），虽然涉及公司但 mentee 没有规范意识 |

### 边界案例

| 消息 | Q4 | 说明 |
|------|-----|------|
| "Why can't I contribute a new article to wikipedia?" | 是 | "can't"=被阻止→事后规范寻求 |
| "Why is this page published on Wikipedia? This page is locked from edits too" | 是 | "locked"=执行结果，质疑内容→规范寻求 |
| "can I make a page for experience (like just a private one no one can see) I wanna practice a little first" | 否 | "can I"=操作问题（"有没有练习区"），无政策词→Q4=否 |
| "I need to take some knowledge about Wikipedia rules" | 否 | 提到"rules"但没有具体规则疑问，泛泛表达学习意愿→Q4=否 |
| "I'll do my best to follow all the rules. I definitely don't want to be a vandal!" | 否 | 提到 rules 和 vandal 但没有在**问**规则。Q0=否（无实质请求） |
| "What to do if we have argument over an edit in wiki? How can it be resolved?" | 是 | 问冲突解决流程→规范寻求 |
| "I have secondary sources to add but when I try to add them on the Talk page, I cannot because it is semi-protected. Where can I post?" | 否 | 被保护限制挡住，但问的是"去哪里发"（操作问题），不是"为什么被限制"→Q4=否 |
| "I'd Be Allowed To Make My Own Wikipedia Meaning For B.B.C" | 是 | "allowed"=政策词→Q4=是 |
| "someone removed and ask for submit proof then how to upload proof" | 是 | "removed"=执行结果→Q4=是。虽然 mentee 主要在问操作（how to upload），但前半句描述了政策执行 |
| "I want to create a biography article of my mentor who is someone I admire" (无政策词) | 否 | 仅提到敏感对象（为他人写传记），但无规范寻求语言→Q4=否。**对比**：同一个人如果说 "...while staying within the rules" → Q4=是 |
| "can I use grammarly to help with the editing?" | 否 | 无政策词→当作操作问题。虽然 Wikipedia 确实有工具使用政策，但 mentee 没有表达合规关切。**对比**："is it **allowed** to use grammarly?" → Q4=是 |
| "Can I create two accounts?" | 否 | 无政策词→当作操作问题。**对比**："Can I create two accounts, or is that **against the rules**?" → Q4=是 |
| "Can I add my own research findings to this article?" | 否 | 无政策词→当作操作问题。**对比**："Can I add my own research, or does that count as **original research**?" → Q4=是 |

---

## Q5 Own Work：用户是否已经做了一些工作？

### 定义
mentee 是否带着**已经在 Wikipedia 上产生了可观察结果的编辑活动**来提问？需要有证据表明编辑行为**已经发生**——内容已保存、已提交、已被他人看到或处理——而非仅处于准备、研究或计划阶段。Q5 反映的是 mentee 的**当前社会化阶段**——是已经积极参与了可以指向的工作，还是从零开始。Q5 不关心 mentor 是否能定位到具体内容（那是 Q3 的事），只关心 mentee 是否已经有了编辑行为或成果。

与 Q3 的区分：
- **Q3**：mentee 请求 mentor **去看并给反馈** → 强调 mentor 的行动
- **Q5**：mentee **是否已经做了工作** → 强调 mentee 所处的状态
- Q5=是 不要求 Q3=是。mentee 可以提到过去的工作但问的是无关问题。

### 判断标准
- 用户是否提到了自己**近期已经做过**的编辑/创建/提交的内容？
- **核心判断**：消息中是否有证据表明编辑行为**已经发生**且**与当前请求相关**？关键看有没有**动作完成或后果出现**的信号。
- ⚠ **必须是用户自己在 Wikipedia 上的编辑行为**——不包括他人的活动，也不包括用户在 Wikipedia 之外的现实世界工作（如出版书籍、为期刊写文章）。
- **内容产出信号词**（→ Q5=是）：was removed / gets messed up / keep getting error / not displaying / accidentally deleted / reverted / was declined / saved / submitted / I created / I edited / my draft / my article——这些都表明用户已经产出或修改了内容。
- **账户状态信号词**（→ 仅凭此不足以判定 Q5=是）：blocked / locked out / cannot edit / restricted / suspended——这些描述的是账户状态，不是内容产出。用户可能在产出任何内容之前就被封禁了。
- **未发生/准备中的信号词**（→ Q5=否）：want to / would like to / interested in / don't know how to / working on / planning to / preparing / researching——这些表明用户处于意向或准备阶段，尚未在 Wikipedia 上产生可观察的编辑结果。"I'm working on updating X" 不等于 "I updated X"：前者可能只是在本地研究、收集资料，后者明确表明编辑已保存
- **久远过去的信号词**（→ Q5=否）：years ago / a long time ago / back in [久远年份] / once before。如果用户仅提到久远过去的编辑活动，而当前请求是关于开始新的事情，那么过去的活动只是背景信息，不算当前的自有工作。
- ⚠ 注意措辞陷阱：前半句看似计划，但后半句暴露已经发生。例如 "I'm wanting to update X but it's instantly been removed"——"been removed" 说明编辑已经做了并被回退 → Q5=是
- ⚠ **冠词/代词规律**：**my/the/this + 内容词**（my article / my draft / the article I wrote）通常暗示已有作品 → Q5=是；**a/an + 内容词**（an article / a draft / a page）通常是泛泛提问，没有暗示已有 → Q5=否。
  - ⚠ **"my page"/"our page" 的歧义**：可能意味着 (a) 用户在 Wikipedia 上创建/编辑的页面 → Q5=是，也可能意味着 (b) **关于**用户或其组织的页面 → Q5=否。根据语境区分：如果用户是页面的主体（而非编辑者），仅凭所有格引用不能确立编辑活动。例如："there are inaccuracies in the article about me" → 用户是页面主体，不是编辑者 → Q5=否；"I edited my page yesterday" → 用户是编辑者 → Q5=是。
  - 例如：
  - "How can I publish **an** article?" → Q5=否（泛泛问流程）
  - "How do I get **my** article published?" → Q5=是（"my article"=已有作品）
  - "How to submit **a** draft for review?" → Q5=否（泛泛问流程）
  - "How to submit **my** draft for review?" → Q5=是（"my draft"=已有）
  - "How long does it take **a** page to get published?" → Q5=否（问一般性时间）
  - "How long does it take for **my** page to get published?" → Q5=是（"my page"=已有）
  - "I've written an article, how do I publish it?" → Q5=是（"I've written"=完成时态，已有）

### 是的例子

| 消息 | Q5 | 说明 |
|------|-----|------|
| "I wrote a draft about X [LINK]" | 是 | 有链接，最明确的own work |
| "I edited the article on Y but it was reverted" | 是 | 指明了具体文章 |
| "My article on Z was tagged for deletion" | 是 | 指明了具体文章 |
| "I created a userbox but it's not displaying right" | 是 | 提到了自己创建的具体对象 |
| "I've made my first wiki article but the name is still saying Draft [LINK]" | 是 | 有链接 |
| "I created my village wiki page called 'Jerrimadugu'" | 是 | 指明了具体页面名 |
| "I updated a pic of Dan Barouch on Wikimedia Commons, but the article has not changed" | 是 | 指明了具体文章和操作 |
| "I submitted my draft" (无标题或链接) | 是 | 虽然没给标题，但mentor可以从用户贡献历史中找到。倾向Q5=是 |
| "I'm trying to add a reference but I keep getting an error" | 是 | "keep getting an error"=已经尝试操作并看到了结果 |
| "I'm wanting to update X but it's instantly been removed" | 是 | 前半句像计划，但"been removed"=编辑已发生并被回退 |
| "I started a draft but I'm not sure if it's ready" | 是 | "started a draft"=已经创建了内容 |
| "I made some changes to an article but I'm not sure they saved" | 是 | "made some changes"=已经操作了 |
| "I was editing an article and accidentally deleted a section" | 是 | "was editing"+具体后果=已发生 |
| "I added an infobox but it's not displaying correctly" | 是 | "added"+看到结果=已发生 |

### 否的例子

| 消息 | Q5 | 说明 |
|------|-----|------|
| "In the article for Ron Sexsmith's 2004 album Retriever I want to add a reference" | 否 | 有具体页面但"I want to add"=还没做，只是计划 |
| "I'm interested in editing history articles" | 否 | 有兴趣领域但没有具体作品 |
| "What are the rules about notability?" | 否 | 问一般规则，没有涉及自己的工作 |
| "How do userboxes work?" | 否 | 问一般操作，没有涉及自己的工作 |
| "I want to create an article about X" | 否 | 还没做，只是计划。没有已有工作 |
| "How do I start editing on Wikipedia?" | 否 | 没有涉及具体工作 |
| "I'd like to write about my company" | 否 | 纯意向，还没动手 |
| "I'm interested in editing the article about X" | 否 | 兴趣/意向，没有动手 |
| "I want to add an infobox but don't know how" | 否 | 不知道怎么做=还没动手，纯提问 |

### 边界案例

| 消息 | Q5 | 说明 |
|------|-----|------|
| "My edits keep getting reverted" | 是 | "my edits"+"reverted"=mentee 做了编辑并被回退→Q5=是 |
| "I've been editing X article" (无具体描述问题) | 是 | 指明了具体文章名，mentor可以去查看 |
| "I've been going a bit ham on editing Wikipedia, and there are a few pages [LINK], [LINK], [LINK]" | 是 | 有链接，有实际编辑行为 |
| "this is my first wikipedia page...in the sandbox" (无链接) | 是 | mentor可以从用户的sandbox找到 |
| "I want to write about a particular steam launch...named 'Antic'" | 否 | 用户还没创建页面，只是描述计划 |

---

## 标注流程总结

```
第一步：判断Q0（实质性提问/请求/贡献意愿？）
  └── 否 → Q2-Q5=N
  └── 是 → 继续标注全部

第二步：如果Q0=是，独立判断Q2、Q3、Q4、Q5
  Q2 Referent：缺任务方向？（是-不知道做什么/否-有明确目标）
  Q3 Appraisal：请求对已有工作的反馈？（是/否）
  Q4 Normative：在问规则/政策？（是/否）
  Q5 Own Work：mentee 是否已经做了工作？（是/否）——反映社会化阶段，不关心 mentor 是否能定位
```

### 常见组合模式

| 组合 | 典型消息 | 含义 |
|------|---------|------|
| Q2-Q4=否, Q5=否 | "How do I add a citation?" / "How to create an article about X?" / "I want to write a biography" | 有明确目标的操作性问题（baseline） |
| Q2=是 | "How do I get started?" / "How to create an article?" (泛泛，无类别) | 缺方向，需引导 |
| Q3=是+Q5=是 | "Can you review my draft [LINK]?" | 求对具体作品的反馈 |
| Q4=是+Q5=是 | "Why was my edit reverted?" | 自己的工作被否定，求解释 |
| Q3=是+Q4=是+Q5=是 | "My draft was declined, what do I fix?" | 求反馈+问规则+有具体工作 |

### 预期Base Rate（基于500条抽样估计）

| 维度 | 预期比例 | 说明 |
|------|---------|------|
| Q0=是 | ~80-85% | 大部分消息有实质请求 |
| Q2 Referent | ~10-12% | 无方向类 + 泛泛创建任务（create article/page 无类别限定） |
| Q3 Appraisal | ~7-9% | 需要已有作品才能求评价 |
| Q4 Normative | ~15-25% | 收紧后：仅当 mentee 文本中有明确的规范寻求语言（去掉了仅提敏感对象的情况） |
| Q5 Own Work | ~20-25% | 已完成并提交的具体页面/草稿/编辑 |
| Q0=是, Q4=否, Q2=否, Q5=否 | ~35-45% | 操作性问题，无政策信号（baseline） |
