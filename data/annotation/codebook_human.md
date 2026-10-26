# Wikipedia Mentor Program — Newcomer Question Classification Codebook

## Overview

This codebook is used for classifying the first message sent by newcomers (mentees) to their mentors in the Wikipedia Mentor Program. Each message requires answering 5 questions (Q0 is a preliminary filter; Q2-Q5 are binary yes/no). Q2-Q5 are not mutually exclusive; a single message can satisfy multiple dimensions simultaneously.

Theoretical basis: Q2-Q4 are grounded in organizational socialization theory's newcomer information-seeking types (Morrison 1993; Miller & Jablin 1991), corresponding to Referent (role orientation), Appraisal (feedback), and Normative (norms) information needs respectively. Different information-seeking types imply that mentors need to adopt different response strategies, making it necessary to distinguish these types when analyzing response effectiveness.

**Coding principles**:
- Only consider the message text itself; do not look at the mentor's reply or subsequent conversation
- Do not infer the user's implicit motivations; only code observable features in the text
- When encountering ambiguous cases, lean conservative (code as no)

---

## Quick Reference Card

Consult this table first when coding; refer to the detailed definitions below if questions arise.

| Code | Question | Values | One-sentence criterion | Keywords/Signals |
|------|----------|--------|----------------------|------------------|
| **Q0** | Substantive question/request/intent to contribute? | Yes/No | Is there something a mentor can respond to? | Has question marks, has "how/what/why", has "I want to..." → lean yes. Pure "Hello"/pure "Thanks" → no |
| **Q2** | Lacks task direction? | Yes/No | Does the user **not know what to do**? | No direction ("where do I start") → yes. Generic creation ("create an article/page" without category) → yes. Has specific category ("biography"/"about someone") → no. Has specific topic → no. Single-step operation ("add image") → no |
| **Q3** | Requesting feedback? | Yes/No | Does the user expect the mentor to **look at specific content** and evaluate it? | "review", "look at", "let me know if", "is this okay" → yes. "help me publish" → no |
| **Q4** | Seeking normative information? | Yes/No | Does the mentee's text **explicitly express seeking information about rules/norms/permissions**? | **Proactive**: "allowed"/"rules"/"policy"/"guidelines"/"violations" → yes. **Policy concept names**: "notability"/"conflict of interest"/"reliable source" etc. → yes. **Reactive**: "blocked"/"deleted"/"reverted"/"declined"/"why was my..." → yes. Only mentions sensitive subject but no norm-seeking language → no. Pure operational steps ("how do I add a heading") → no. ⚠ "best practice"/"recommended" etc. do not automatically trigger Q4 — check whether mentee is asking about **rules/standards** or **practical advice** |
| **Q5** | Has already done work? | Yes/No | Does the mentee **come with editing activity that has produced observable results on Wikipedia**? | "my draft", "my article", "I edited/created...", "I have a draft ready" → yes. "I want to create..."/"I'm working on..." → no (planning/in progress ≠ existing work) |

**Coding order**: Q0 → If Q0=yes then code Q2/Q3/Q4/Q5, otherwise Q2-Q5=N

**Quick decision flowchart**:

```
1. Read the message and ask yourself: What can a mentor respond to?
   → Nothing to respond to (pure greeting/pure thanks/unrelated content) → Q0=no, Q2-Q5=N
   → Has question/request/intent to contribute → Q0=yes, continue

2. Q2: Does the user know what they want to do?
   → Completely directionless ("where to start"/"what to edit") → yes
   → Generic creation task ("create an article"/"make a page", no category specified) → yes
   → Has specific category ("biography"/"about an artist"/"add someone") → no
   → Has specific topic ("article about X") → no
   → Single-step operation ("add image"/"add citation"/"fix error") → no

3. Q3: Has the user said "please look at my work" and give feedback?
   → Yes (review/check/look at + requesting feedback) → yes
   → Only requesting help doing something (publish/move/approve) → no

4. Q4: Is the mentee **seeking normative/rule information**?
   → Proactive: has policy words (allowed/rules/policy/permitted/guidelines/violations) or permission questions ("is it ok to"/"am I allowed") → yes
   → Reactive: work affected by policy enforcement (blocked/deleted/reverted/declined/rejected/removed/warned) or asking why ("why was my..."/"why can't I...") → yes
   → Only mentions sensitive subject (person/company/self) but without the above norm-seeking language → no
   ⚠ Distinguishing from Q2: "how to create a page" → Q2=yes, Q4=no; "am I allowed to create a page" → Q4=yes

5. Q5: Has the mentee already done work? (reflects socialization stage)
   → Evidence that editing has occurred on Wikipedia (my draft/I edited/was reverted/I created) → yes
   → Plans, intentions, or preparation (I want to/I'd like to/I'm working on) → no
```

---

## Q0: Does this message contain a substantive question or request related to Wikipedia editing?

### Definition
Can the user's message enable the mentor to provide a **substantively differentiated reply**? If the mentor can only respond with a generic welcome template or "this isn't a Wikipedia issue," then Q0=no. If the mentor can provide tailored guidance based on the user's specific situation, then Q0=yes.

### Decision Criteria
Core criterion: **Would the mentor's reply meaningfully differ because of the content of this message?**

**Q0=yes** — Messages where the mentor can give a targeted reply:
- Has a question or request about Wikipedia editing/contributing
- Has a specific question about Wikipedia account/process/community (including username, password recovery, dark mode, etc. — as long as it is related to the Wikipedia platform)
- Expresses clear intent to edit/contribute (even without a specific question)

**Q0=no** — Messages where the mentor can only give a generic reply, complete list:
1. **Pure greeting**: "Hello!" / "Hi!" / "Nice to meet you!" (with no additional content)
2. **Pure thanks**: "Thanks!" / "Thank you for the welcome!"
3. **Too short/meaningless**: Single words, garbled text, incomprehensible content
4. **Not Wikipedia-related**: Facebook/Gmail/Instagram/WiFi passwords, phone issues, Google search problems, job-seeking, whether Wikipedia pays editors, etc.; also includes requests for content types that Wikipedia does not host (Wikipedia is an encyclopedia; requests that misunderstand its nature belong here)
5. **Non-English**: The main body of the message is not in English
6. **Asking if it's a bot**: "Are you a real person?" / "Are you a bot?" (with no other substantive content)
7. **Sharing experiences/feelings unrelated to Wikipedia editing with no specific question**: "It is my first day to join Wikipedia and I am so excited!" / "I have no questions for now" (pure sharing, no editing/contributing intent, no request. Note: if the user mentions their own editing behavior or intent to contribute, e.g. "Put in my first edit today!", this is Q0=yes, as it expresses clear editing/contributing intent)
8. **Pure self-introduction with no request**: "hello I am a 10 y/o kid" / "I am Paul"


⚠ Note: The following may seem "non-editing" but are actually Q0=yes:
- **Username issues**: Involves attribution policy; mentor provides substantive guidance → Q0=yes
- **Password/account recovery**: Involves account policies and procedures; mentor gives specific advice → Q0=yes
- **Dark mode/interface settings**: Mentor teaches specific operational steps (gadgets/CSS) → Q0=yes
- **User page editing**: Part of Wikipedia community participation → Q0=yes

### Yes Examples

| Message | Q0 | Explanation |
|---------|-----|------------|
| "How do I add a citation to an article?" | Yes | Operational question |
| "My draft was declined, what happened?" | Yes | Has a specific question, involves rules |
| "Can you look at my article?" | Yes | Has a specific request (review) |
| "I want to start writing articles, how do I begin?" | Yes | Has a specific question (how to start) |
| "How do I change my signature?" | Yes | Wikipedia platform operation |
| "How do I change my username?" | Yes | Involves attribution policy; mentor provides substantive guidance |
| "How do I activate dark mode?" | Yes | Mentor teaches gadgets/CSS specific operations |
| "I forgot my password, how to recover my account?" | Yes | Involves Wikipedia account policy and recovery process |
| "Fix my userpage" | Yes | User page editing = Wikipedia community activity |
| "I'm new here and excited to contribute!" (no specific question) | Yes | Expresses clear intent to contribute = implicit guidance request |
| "Can a Wikipedia page be created about me without my consent?" | Yes | Article subject asking about rules (consent = permission issue) |
| "How do I get my page removed?" | Yes | Article subject requesting modification to Wikipedia content, with a specific page |
| "Someone wrote false information about me, how do I get it taken down?" | Yes | Article subject requesting correction, relates to Wikipedia content |
| "Put in my first edit today! Excited to venture forth" | Yes | Mentions own editing behavior → has clear intent to contribute |

### No Examples

When Q0=no, Q2-Q5=N.

| Message | Q0 | Explanation |
|---------|-----|------------|
| "Hello! Nice to meet you!" | No | ① Pure greeting |
| "Thank you for the welcome message!" | No | ② Pure thanks |
| "No" / "W" / "Earthh" | No | ③ Too short/meaningless |
| "What is energy?" | No | ④ Not Wikipedia-related (knowledge question) |
| "Facebook locked" / "Instagram password cracking" | No | ④ Not Wikipedia-related (other platforms) |
| "How can I get the Wi-Fi password" | No | ④ Not Wikipedia-related |
| "Does Wikipedia pay editors?" / "how do I start earning" | No | ④ Not Wikipedia-related (Wikipedia does not pay) |
| "I'm trying to put my information on Google" | No | ④ Not Wikipedia-related (Google issue) |
| "Are you a real person?" (no other content) | No | ⑥ Asking if it's a bot |
| "It is my first day to join Wikipedia and I am so excited" | No | ⑦ Sharing feelings, no editing/contributing intent, no specific question |
| "hello I am a 10 y/o kid" (no follow-up question) | No | ⑧ Pure self-introduction with no request |
| "free XVideos red" | No | ③ Meaningless content |

### Borderline Cases

| Message | Q0 | Explanation |
|---------|-----|------------|
| "Hello, I created an account only because I thought that I was required to do so in order to use the site. Am I correct?" | No | Asking whether an account is required; does not involve editing activity |
| "I don't understand Wikipedia" | No | Too vague; no specific actionable request |
| "I'm new here and excited to contribute! But this UI is really confusing, I don't know where to start" | Yes | "don't know where to start" = substantive request |
| "I am makw same mistakw in this web sob forgive me i will not occur this in future" | No | Apology; no question or request |

---

## Q2 Referent: Does the user lack task direction — not knowing what to do or where to start? (Yes = lacks direction, No = has direction)

### Definition
The user **lacks task clarity** — does not know what to do, where to focus their effort, or where to start. This corresponds to Morrison 1993's Referent information seeking: newcomers seeking information about role and task requirements.

### Decision Flowchart

```
1. Does the user have no direction at all?
   → "how to start editing" / "where to begin" / "what should I edit" → Q2=yes
   → Has a task they want to do → continue to step 2

2. Does the task description have a specific category or topic?
   → Has specific topic ("article about Odin's Valkyries") → Q2=no
   → Has specific category ("biography" / "about an artist") → Q2=no (came with a specific purpose)
   → Only used generic terms (article / page / wiki) → continue to step 3

3. Generic task vs. single-step operation?
   → create/write/make + article/page/wiki → Q2=yes (too generic, essentially indistinguishable from lacking direction)
   → add image / add citation / fix error etc. → Q2=no (single-step operation)
```

### Complex Task List vs. Single-Step Operation List

Coders do not need to judge task complexity themselves; consult the list directly:

**Generic creation tasks (Q2=yes)** — These expressions are too generic; the user has not expressed any specific direction, and they are essentially indistinguishable from "don't know where to start":
- create/write/make/start/add + **article** / **page** / **wiki** / **entry** / **new page** / **new article**
- Key characteristic: only used the most generic verb + most generic object (article, page, wiki), with no category specification
- ⚠ Note: **post/publish/submit + article/page** do not belong to this category → Q2=no. These verbs imply the user already has content and is asking about the publishing process, which is a single-step operation

**Creation tasks with specific categories (Q2=no)** — Once the user specifies a content type or object category more specific than "article/page/wiki," it indicates they came with a specific purpose and do not lack direction. Decision rule: **One level more specific than article/page counts as having direction.** Common categories include:
- **People**: biography / profile / someone / a person / "add someone to Wikipedia"
- **Organizations**: company / organization / business / brand / agency / church
- **Film/Culture**: film / movie / TV show / series / album / song / band / book / novel
- **Places**: place / city / town / village / country / school / university
- **Sports**: team / player / match / athlete
- **Events**: event / incident / battle / war
- **Wikipedia-specific types**: template / userbox / user page / list / redirect / disambiguation / category page / draft / essay / portal / stub / infobox / help page / talk page / WikiProject page / file page / media page
- **Domain-specified**: write about **a topic I know** / contribute on **AI/technology** / edit **history articles**

**Single-step operations (Q2=no)** — These tasks are self-contained operational instructions; the mentor can directly teach the steps:
- **post** / **publish** / **submit** + article / page / draft (already has content, asking about publishing process)
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

### Decision Criteria Summary
- **Completely directionless** ("where to start"/"what to edit") → Q2=yes
- **Generic creation task** (create/write + article/page/wiki, no category specified) → Q2=yes
- **Has specific category** (biography/profile/company page/about an artist) → Q2=no
- **Has specific topic** ("article about Odin's Valkyries") → Q2=no
- **Single-step operation** (add image/citation/infobox etc.) → Q2=no
- **Already encountered a specific obstacle while editing** → Q2=no
- **"don't know where to start"**: if there is a specific topic, code as Q2=no ("don't know where to start" = asking about process); if there is no specific topic, code as Q2=yes
- If "any advice"/"any suggestions" is the **core request** of the message → Q2=yes; if it comes as a **polite closing** after describing a specific problem → Q2=no

### Yes Examples

| Message | Q2 | Explanation |
|---------|-----|------------|
| "How can I start my editing on Wikipedia?" | Yes | Classic Referent: doesn't know where to start |
| "What should I edit? I know a lot about computers but I don't know what to edit" | Yes | Has ability but lacks direction; the purest Referent |
| "Which kind of pages should I edit and which kind of pages should I avoid?" | Yes | Seeking priorities and boundaries |
| "where do you start when it comes to editing? is there a recommended place to start?" | Yes | Explicitly seeking direction |
| "I would like to translate articles from English to my language. But I don't know where to start" | Yes | Has a goal but lacks a path |
| "Can you mentor me through this whole process. I would really appreciate." | Yes | Seeking full-process guidance |
| "Hello, teach me how to edit" | Yes | "teach me" implies a need for systematic learning |

### No Examples

| Message | Q2 | Explanation |
|---------|-----|------------|
| "I am new to Wikipedia, but have long time interest in Caucasus regional politics. I want to write about the regional politics and started with one of the politician — I am not sure if I should proceed with other pages or just wait for approval" | No | Has a specific project but is asking about strategy (continue or wait) |
| "How do I create an article about Odin's Valkyries?" | No | Has specific topic → Q2=no |
| "I want to create a biography" | No | "biography" = specific category → Q2=no |
| "Hello, please how do you add someone to wikipedia?" | No | "add someone" = creating a person's page, has specific category (person) → Q2=no |
| "How do I add an infobox?" | No | Single-step operation list → Q2=no |
| "How do I add a citation?" | No | Single-step operation list → Q2=no |
| "How do I add a link to the references section?" | No | Single-step operation list → Q2=no |
| "How do I change my signature?" | No | Single-step operation list → Q2=no |
| "How to i post a wiki" | No | "post" = publishing operation, implies existing content → Q2=no |
| "I'm interested in history, any suggestions for what to work on?" | No | Has an interest area (history = specific category) → Q2=no. "any suggestions" is searching for specific topics within that area, not lacking direction |
| "Can you review my draft?" | No | Has specific work; seeking evaluation, not direction |
| "Why was my edit reverted?" | No | Not seeking direction; asking about rules |
| "Please move my page to mainspace" | No | Requesting execution, not seeking direction |
| "is it recommended to use less citations or more citations for short articles?" | No | Asking about community conventions/best practices, not lacking direction |

### Borderline Cases

| Message | Q2 | Explanation |
|---------|-----|------------|
| "Hello am I wanna know how to edit my Wikipedia" | Yes | No specific goal/topic → Q2=yes |
| "Any advice you have re editing or creating Wikipedia entries most welcome" | Yes | Generic "any advice" as the main request of the message → Q2=yes |
| "how to create an article about Odin's Valkyries but I do not know where to start" | No | Although says "don't know where to start," has a specific topic (Odin's Valkyries) → Q2=no. "don't know where to start" when a specific topic exists = asking about process, not lacking direction |
| "I went to the page for Joel Hurt and added his grandchild...I'm open to any advice" | No | "any advice" follows a specific problem description; it is a polite closing. User has a clear goal |
| "What can I do about this?" (edits repeatedly reverted by others) | No | "what can I do" points to a solution for a specific predicament, not lacking direction |

---

## Q3 Appraisal: Is the user requesting feedback or evaluation on their existing work?

### Definition
The user has already done some work (written a draft, made an edit, built a page), and is now asking someone to **look at their work and provide a judgment** — how is it, what needs to change, whether it is good enough.

### Decision Criteria
Core criterion: **Does the user expect the mentor to look at specific content and provide evaluation/feedback/improvement suggestions?**
- "review", "check (and give feedback)", "look at...let me know", "is this okay", "how can I improve" → Q3=yes
- "help me publish", "approve my draft", "when will it be approved" → Q3=no (requesting an action or information, not evaluation)
- ⚠ **Publishing/submission guidance for specific existing work**: When the user seeks publishing/submission **guidance or advice** (advise, best way to proceed, how should I) regarding a specific existing article, determine: would the mentor need to open and read the user's content to provide a meaningful answer? If yes → Q3=yes (implicit feedback request); if the mentor only needs to give general process steps to answer → Q3=no. For example: "Please advise on the best way to proceed with publishing this article" (seeking guidance on a specific article) → Q3=yes; "How do I publish an article?" (generic process question) → Q3=no

### Yes Examples

| Message | Q3 | Explanation |
|---------|-----|------------|
| "Can you review my article in draft and let me know if it will be moved to the main space?" | Yes | Explicitly requests review |
| "Can you take a look at the article I created and let me know if there is anything that I need to fix?" | Yes | Seeking feedback + improvement suggestions |
| "I have formulas and figures in my drafts. Are there problems?" | Yes | Seeking work inspection |
| "Please can you help check this article out and let me know how I can improve it? I've been working on it for a while now." | Yes | Seeking improvement suggestions |
| "I've been going a bit ham on editing Wikipedia, and there are a few pages where I made some pretty hefty edits. If you have some time and would be willing to take a look...it would be great to know whether I'm headed in the right direction" | Yes | Seeking direction confirmation |
| "I've created a draft that I would like to publish. Would you be able to review it?" | Yes | "review" = look and give feedback |
| "I created a page. Please take a look and let me know if everything is done the right way" | Yes | Seeking confirmation |
| "this is my first wikipedia page...in the sandbox. Please let me know if this is fine so far" | Yes | Seeking feedback |
| "for my article I get the feeling that more citations are better but not sure if that is the case: [LINK]" | Yes | Bringing specific work and asking about best practices |
| "I look forward to your critique and suggestions on how I can improve" | Yes | Explicitly seeking critique |
| "I'd really like some advice on my contribution to the Mad Max: Fury Road page" | Yes | Seeking feedback on own contribution |
| "I've been working on it for 2 months but didn't get response, only response I got is declined. I don't know what to change on that article, can you help me out with this?" | Yes | "don't know what to change" = needs someone to look and tell me |
| "Can you kindly review my paid editing disclosure and confirm it's compliant?" | Yes | Seeking compliance review |
| "Can you assist to verify the content before publish?" | Yes | Seeking review |
| "I posted my first page on Sunday...Can you please review the posting and kindly let me know what needs to be done to have it accepted?" | Yes | Explicitly requests review + seeks improvement direction |
| "I've enjoyed completing the tasks...When and how should I decide whether to remove the issues tags? See for example Alan Bircher: [LINK] — I added more thorough documentation..." | Yes | "See for example" + link + describes changes = implicitly asks mentor to look and judge |

### No Examples

| Message | Q3 | Explanation |
|---------|-----|------------|
| "I've Drafted an article! Can you help me Publish it?" | No | Requesting help executing publication; no indication of seeking feedback |
| "When will I be able to submit my draft for review?" | No | Asking about process/timing |
| "How do I add images to my draft?" | No | Operational question |
| "Where can I turn my draft to a public article?" | No | Asking about process |
| "how long does it take a page to get published?" | No | Asking about process |
| "How do I publish an article?" | No | Operational question |
| "could you advise when my article will be approved for publishing?" | No | Asking about timing, not seeking evaluation |
| "Please approve My draft" | No | User thinks the mentor has approval authority; requesting execution of an action |
| "How do I know if the article I edited has improved?" | No | Asking about evaluation methods/standards, not requesting someone to evaluate. If they said "has my article improved?" then Q3=yes |
| "Any chance you're willing to make some edits so this is strong enough to submit?" | No | Requesting the mentor to make edits (Service), not seeking feedback |
| "on my edits there are numbers like (+118)...what do those mean?" | No | Asking about interface features |
| "I got warned cos my editing was shyt" | No | Reporting an event; not requesting feedback |
| "Any advice you have re editing or creating Wikipedia entries most welcome" | No | Generic advice-seeking, not feedback on specific work. Q2 is more appropriate |

### Borderline Cases

| Message | Q3 | Explanation |
|---------|-----|------------|
| "am I doing well? 44 edits not bad" | Yes | Seeking overall performance evaluation, though not pointing to specific work |
| "you noted that my contribution lacked notability. I disagree..." | No | Responding to existing appraisal and seeking reassessment; more like disputing/appealing |
| "I created an account to edit an Article and create one about Marcel Bonin. I know that Wikipedia needs sources and I wondered how all that works" | No | Though mentions own project, the question is about operational methods (how to add sources), not seeking evaluation |
| "The article is quite ready to be published...I need to do at least 10 reviews to publish. Am I right?" | No | Asking about process rules, not seeking evaluation of the article |

---

## Q4 Normative: Is the mentee seeking normative/rule information?

### Definition
Does the mentee's message text **explicitly express seeking information about rules, norms, or permissions**? Q4 codes the mentee's information-seeking behavior (Morrison 1993 Normative information seeking), not speculation about whether the mentor's reply would involve policy.

Q4's coverage is consistent with Morrison 1993's definition of normative information — encompassing all aspects of expected behavior within the organization, not limited to formally enforced policies. Wikipedia's normative system has four tiers, and Q4 covers all of them:
- **Policies** (notability, BLP, COI) → Enforced
- **Guidelines** (reliable sources, NPOV) → Have enforcement power
- **Manual of Style** (heading format, punctuation conventions) → Community-consensus formatting norms
- **Essays/conventions** (paragraph length, citation quantity preferences) → Community best practices

**Core principle**: Only look at what the mentee said, not whether the object the mentee mentions objectively involves policy. Even if the mentee mentions living persons / companies / self and other sensitive subjects, as long as the mentee's language contains no norm-seeking signals, code Q4=no.

### Decision Criteria
Two types of textual signals (at least one type present → Q4=yes; neither present → Q4=no):

**Type A: Proactive Normative Seeking** — The mentee proactively asks about rules/permissions/compliance before acting:
- **Norm-seeking language** (each word with typical context):
  - **allowed/permitted**: "Am I allowed to edit this page?" / "Is it permitted to use YouTube as a source?"
  - **rules/policy/guidelines/terms**: "What are the rules for creating a new page?" / "I want to follow the correct guidelines"
  - **violations/violating**: "I don't want to violate any Wikipedia terms" / "Is this a violation?"
  - **consent/authorize**: "Can a page be created without my consent?" / "I didn't authorize this page about me"
  - **permission**: "Do I need permission to edit a protected page?" / "Who gives permission to publish?"
  - **prohibited/banned/blocked**: "Is it prohibited to edit my own company's page?" / "Will I get banned for this?" / "Why am I blocked?"
  - **appropriate/legitimate**: "Is it appropriate to add my own research?" / "Is this a legitimate source?"
  - **ok/okay** (in compliance context): "Is it ok to use images from Google?" / "Is it okay if I edit a page about a living person?"
  - **qualify/eligible/criteria**: "Does this topic qualify for a Wikipedia page?" / "Am I eligible to create articles?" / "What are the criteria for notability?"
  - **compliant/compliance**: "Is my disclosure compliant?" / "I want to make sure I'm in compliance with Wikipedia's policies"
  - **legal/illegal**: "Is it legal to copy text from another website?" / "Would it be illegal to use this image?"
  - **proper/properly** (in compliance context): "How to properly cite sources without getting in trouble?" / "What is the proper way to disclose paid editing?" (Note the distinction: if "properly" only refers to operational correctness such as "how to properly format a table" → Q4=no)
- **Implicit compliance inquiry (self-flagging pattern)**: The mentee does not directly use policy words, but **proactively lists reasons why their content might not meet standards**, then asks "can I do it/can I publish it." This "first flag potential issues → then ask if it's feasible" structure shows the mentee is aware of compliance risk and is essentially asking "is this type of content accepted by Wikipedia." Key points:
  - ✅ Q4=yes: "there is no official proof" / "it is completely fantasy" / "only written on paper" + "will this be possible to publish?" — the mentee proactively provides factors that might violate notability/verifiability; they are not asking "how to publish" (operational) but "can this type of content be published" (compliance)
  - ✅ Q4=yes: "he's not very well known and there aren't many sources about him, can I still make a page?" — Proactively noting insufficient sources and asking if it's feasible
  - ❌ Q4=no: "I want to create a page about my fantasy wrestling league, how do I start?" — Does not list any potential issues; pure operational question
  - **Key distinction**: Simply asking "can I do X" → operational question → Q4=no; asking "can I do X" while **proactively stating reasons why X might not meet standards** → compliance inquiry → Q4=yes
- **Permission thresholds/entry requirements**: "how many edits to get privileges" / "when can I create my own pages" / "confirmed user" / "how many/much X do I need to pass/be allowed/get accepted" (when the sentence structure is "how much X do I need to be allowed/pass Y," it indicates the mentee is asking about a standard or threshold, which is normative seeking)
- **Permission scope/boundary inquiry**: "what type of X can I put/add/share" / "which type of links are allowed" — when the mentee asks "what type of content/links/information can be added," they are essentially asking what is permitted and what is not, which is an inquiry into normative boundaries. Distinction: if asking "how to add a link" (how to add a link) → pure operation → Q4=no; if asking "what kind of links can be added" (what type of links to add) → asking about permission scope → Q4=yes
- **Policy concept names** — When the mentee is **asking about or discussing** the rules/standards/applicability of a Wikipedia policy concept → Q4=yes. Note: merely mentioning a policy concept name as a known term to describe something else does not count. For example: "What counts as a reliable source?" → asking about rules → Q4=yes; "I found a reliable source and want to add it" → descriptive use → Q4=no. "What counts as vandalism?" → asking about rules → Q4=yes; "What pages do you find most frequently vandalized?" → using vandalism to describe direction → Q4=no. Common policy concept names:
  - Content standards: "notability" / "notable enough" / "reliable source" / "verifiability" / "neutral point of view" / "NPOV" / "original research"
  - Interest-related: "conflict of interest" / "COI" / "paid editing" / "disclosure"
  - Copyright: "copyright" / "fair use" / "public domain" / "free license"
  - Behavioral norms: "edit war" / "three-revert rule" / "3RR" / "vandalism" / "sock puppet" / "canvassing"
  - Process-related: "speedy deletion" / "AfD" / "articles for deletion" / "semi-protected" / "autoconfirmed"

**Type B: Reactive Normative Seeking** — The mentee's work has been affected by policy enforcement, and they are seeking to understand why:
- **Enforcement result words**: "blocked" / "deleted" / "reverted" / "declined" / "rejected" / "removed" / "warned" / "undone" / "locked" / "semi-protected" / "not published" / "not incorporated" / "not accepted" / "disappeared"
- **Requests centered on enforcement results** → Q4=yes, including all of the following:
  - (a) Explicitly asking why: "why was my..." / "why can't I..." / "why did they..." / "what happened to my..."
  - (b) Asking how to resolve or undo: "how do I fix this" / "how can I get it back"
  - (c) Expressing confusion: "what happened?" / "what shall I do?" / "what am I doing wrong?"
  - (d) Only reporting the enforcement result without other requests — the implied question is still "why/what should I do"
  - (Resolving any policy enforcement result necessarily requires the mentor to explain the underlying rules or norms that triggered it.)
  - Note: Q4=yes here is unrelated to Q3. A message can **simultaneously** seek feedback on work (Q3=yes) **and** involve normative issues (Q4=yes).
- **Disputing/appealing**: "I would like to contest..." / "I disagree with..." / "how is this fair..."
- ⚠ **Stating background vs. centering on enforcement result**: When enforcement result words appear, also check whether the mentee's **main request centers on the enforcement result itself** or **pivots to a completely different request**. For example: "My first article was deleted (I was not done)...Can you look at my new article?" — the mentee mentions "deleted" but pivots to a new request (review) → judge Q4 separately for the pivoted request. Compare: "My article was deleted, why?" → Q4=yes

**Q4=no**: The mentee's text contains none of the above norm-seeking signals. The following situations are coded as no even if they would objectively trigger a policy-related reply:
- Only mentions sensitive subjects ("create a page about a politician" / "write about my company") without norm-seeking language and without proactively listing potentially non-compliant factors
- Pure operational questions ("how do I create a page about X")
- "Can I" + no policy words, no compliance context, no self-flagging → treated as operational question
- ⚠ **Mentioning a process/policy name ≠ seeking normative information**: The mentee mentions a Wikipedia process or policy concept name, but is actually asking about **operational steps** rather than **the rules themselves**; still coded Q4=no. Key distinction: is the mentee asking "how to operate" or "what the rules are / whether it is allowed." For example:
  - "How do I submit through Articles for Creation?" → asking about operational steps (can't find the button) → Q4=no
  - "What are the criteria for Articles for Creation to accept my draft?" → asking about review standards/rules → Q4=yes
  - "How do I submit for speedy deletion?" → asking about operation → Q4=no
  - "What qualifies for speedy deletion?" → asking about rules → Q4=yes
- ⚠ **Operational keywords easily confused with normative** (all Q4=no):
  - "what formats are accepted" → technical file format specifications, Q4=no
  - "how do I create/make/start" → operational steps, Q4=no
  - "can I delete/undo/revert" → operational capability question, Q4=no
  - "how many articles can we write" → factual platform capability question, Q4=no
  - "what do I have to do before X is possible" → process steps, Q4=no
- ⚠ **"best practice"/"recommended"/"should I" do not automatically trigger Q4**: These words often appear in contexts seeking practical advice; the mentor's answer is experience-based operational advice rather than citing policy. For example:
  - "is it recommended to use less citations or more?" → asking practical advice, Q4=no
  - "how many citations should I add?" → asking practical advice, Q4=no (no Wikipedia policy specifies citation quantity)
  - "what is the best practice for sourcing?" → asking practical advice, Q4=no
  - Compare: "is it recommended **per Wikipedia policy** to use less citations?" → explicitly references policy → Q4=yes

### Distinguishing from Q2
Q2 (path planning) and Q4 (normative seeking) can both be yes simultaneously. Q2 looks at sense of direction; Q4 looks at norm-seeking language:

| Message | Q2 | Q4 | Basis |
|---------|----|----|-------|
| "How to create a page?" | Yes | No | Generic creation → Q2=yes. No norm-seeking language → Q4=no |
| "How do I create an article about X?" | No | No | Has topic → Q2=no. No norm-seeking language → Q4=no |
| "How do I create a page about a politician?" | No | No | Has specific category → Q2=no. No norm-seeking language → Q4=no |
| "create a page for my company" | No | No | Has specific category → Q2=no. No norm-seeking language → Q4=no |
| "Am I **allowed** to create a page about a politician?" | No | Yes | "allowed" = norm-seeking → Q4=yes |
| "What type of information can I share on my user page?" | No | Yes | "what type of X can I share" = asking about permission scope → Q4=yes |
| "How do I add a link to my user page?" | No | No | "how do I add" = pure operational question → Q4=no |
| "Which type of links can I add on my user page?" | No | Yes | "which type of X can I add" = asking what types are allowed → Q4=yes |
| "I want to create my own biography, is that **against the rules**?" | No | Yes | "against the rules" = norm-seeking → Q4=yes |
| "How to add an image?" | No | No | Single-step operation → Q2=no. No norm-seeking language → Q4=no |
| "Am I **allowed** to add an image?" | No | Yes | "allowed" = norm-seeking → Q4=yes |
| "My edit was **reverted**, why?" | No | Yes | "reverted" + "why" = reactive norm-seeking → Q4=yes |

### "Can I do X" Decision Rules
"Can I" in English can mean either "technically able to" or "permitted to by rules." Basis for judgment:
- "Can I" + **policy words/compliance context** ("allowed", "permitted", "without getting in trouble", "conflict of interest", "copyright" etc.) → Q4=yes
- "Can I" + **no policy words, no compliance context** → Q4=no, treated as operational question
- ⚠ Do not infer the meaning of "Can I" based on whether the object is sensitive — the mentee must express compliance concern themselves through language

**Typical "Can I" Q4=no examples** (pure operation/technical capability/platform features):

| Message | Q4 | Explanation |
|---------|-----|------------|
| "Can I edit a page on my phone?" | No | Asking about technical capability |
| "Can I add bold text to an article?" | No | Asking about formatting operation |
| "Can I undo my last edit?" | No | Asking about operational steps |
| "Can I save a draft and come back later?" | No | Asking about platform features |
| "Can I see who edited a page?" | No | Asking about platform features (edit history) |
| "Can I edit a page without creating an account?" | No | Asking about platform capability (anonymous editing) |
| "Can I translate an article from another language?" | No | Asking about operational process |
| "Can I create a page about my professor?" | No | Only sensitive subject, no policy words |
| "Can I write about my company?" | No | Same as above |
| "Can I make my own biography?" | No | Same as above |

**Same object, Q4=no vs. Q4=yes comparison**:

| Q4=no (no policy words) | Q4=yes (has policy words) | Difference |
|--------------------------|---------------------------|------------|
| "Can I create a page about my company?" | "Can I create a page about my company without it being a **conflict of interest**?" | Added policy concept name |
| "Can I write about myself?" | "Is it **allowed** to write about myself?" | Added policy word |
| "Can I add a photo I found online?" | "Can I add a photo I found online, or is that **copyright** infringement?" | Added policy concept name |
| "Can I edit the page about my company?" | "Can I edit the page about my company, or would that **violate** some policy?" | Added policy word |
| "Can I create two accounts?" | "Can I create two accounts, or is that **against the rules**?" | Added policy word |
| "Can I add my own research findings?" | "Can I add my own research, or does that count as **original research**?" | Added policy concept name |

### Yes Examples — Type A: Proactive Normative Seeking

The mentee proactively asks about rules/permissions/compliance before acting.

| Message | Q4 | Explanation |
|---------|-----|------------|
| "Am I allowed to put in my own opinions?" | Yes | "allowed" = directly asking if permitted |
| "I would like to understand why I cannot upload an image of a public figure I found on the internet. Is there a way to use this image under the fair use policy?" | Yes | "fair use policy" = explicitly mentions policy |
| "I'm still quite confused on the uploading an image rules" | Yes | "rules" = explicit policy word |
| "Can I use YouTube links as a reference?" | Yes | "as a reference" = asking what source types are accepted → permission scope inquiry (similar to "what type of sources can I use") |
| "How can I know when I'm a confirmed / auto-confirmed user to start my own pages?" | Yes | Asking about permission thresholds |
| "how many edits does it take to have editing privileges?" | Yes | "privileges" = permission rules |
| "I'm being accused of 'gaming' the system. Thoughts?" | Yes | Asking about behavioral norms |
| "I think I broke the three edit war rule" | Yes | "rule" = policy word |
| "how to properly edit a Wikipedia article without any hassle and avoid any potential violations" | Yes | "violations" = explicit policy word |
| "is it allowed to re-do an entire article given to you for editing for its betterment?" | Yes | "allowed" = directly asking if permitted |
| "if I find a subject without a page, can I take the liberty and create a quality page?" | Yes | "take the liberty" = uncertain whether permitted → compliance awareness |
| "How to reliably edit so my edits won't be dismissed as 'Lies'?" | Yes | "dismissed" = worried about policy enforcement rejection |
| "I would like to follow the correct guidelines so I don't violate any Wikipedia terms" | Yes | "guidelines" + "violate" + "terms" = multiple policy words |
| "I wanted to create a wiki page for the musician 'she hacked you'...How best do I approach this so that I follow the rules?" | Yes | "follow the rules" = explicit norm-seeking |
| "I want to create a biography article of my mentor...while staying within the rules of providing independent, third-party references" | Yes | "within the rules" + "independent, third-party references" = explicit norm-seeking |
| "Am I allowed to create a page about a politician?" | Yes | "allowed" = policy word → Q4=yes (Note: without "allowed," just saying "How do I create a page about a politician" → Q4=no) |
| "What are the notability requirements for a Wikipedia article?" | Yes | "notability" = policy concept name → Q4=yes |
| "Do I have a conflict of interest if I edit my company's page?" | Yes | "conflict of interest" = policy concept name → Q4=yes |
| "What counts as a reliable source?" | Yes | "reliable source" = policy concept name → Q4=yes |
| "Is this copyright infringement if I use an image from Google?" | Yes | "copyright infringement" = policy concept name → Q4=yes |
| "I submitted my first draft and was declined because I cited incorrectly...the article has notability due to press coverage" | Yes | "declined" = enforcement result + "notability" = policy concept name → Q4=yes |
| "I want to properly disclose any conflict of interest to make sure everything is transparent" | Yes | "conflict of interest" + "disclose" = policy concept name + compliance awareness → Q4=yes |

### Yes Examples — Type B: Reactive Normative Seeking

The mentee's work has been affected by policy enforcement (deletion/reversion/rejection/blocking etc.), and they are seeking to understand the reason.

| Message | Q4 | Explanation |
|---------|-----|------------|
| "why wikipedia removing my editing lines" | Yes | "removing" = enforcement result + "why" = asking the reason |
| "why are people downvoting my edits?" | Yes | "downvoting" = enforcement result + "why" = asking |
| "it is rejected by Wikipedia because I personally didn't write the ad back in 1927...How are these images allowable while mine is not?" | Yes | "rejected" + "allowable" = questioning rule consistency |
| "I edited the fordite wiki and the edit was deleted...The current wiki contains several false and misleading statements" | Yes | "deleted" = enforcement result |
| "one of my pages has been deleted. I would like to know what was wrong with it" | Yes | "deleted" + "what was wrong" = asking the reason |
| "I would like to contest the speedy deletion of the page because I feel that it was wrongfully deleted" | Yes | "deletion" + "contest" = disputing/appealing |
| "rejected with the explanation that it constituted advertising...I invested considerable time ensuring the content adhered to Wikipedia's guidelines" | Yes | "rejected" + "guidelines" = reactive norm-seeking |
| "All the contributions I have made were reverted. I have not changed the text; I just added a citation..." | Yes | "reverted" = enforcement result, implying "why" |
| "My athlete page Barry Nobles was deleted and I can't seem to find out why" | Yes | "deleted" + "why" = asking the reason |
| "Can you help me submit this correctly? I'm planning to submit a new version now after the first version was declined" | Yes | "declined" = enforcement result |
| "Why cannot I add some of the links in the references?" | Yes | "cannot" = being blocked, reactive norm-seeking |
| "two of my articles were declined of lack of references" | Yes | "declined" = enforcement result |

### No Examples — Operational Questions (no norm-seeking language)

| Message | Q4 | Explanation |
|---------|-----|------------|
| "How Can I publish my article?" | No | Operational question |
| "How can I edit my page?" | No | Operational question |
| "How can I add an article?" | No | No norm-seeking language → Q4=no |
| "Any tips on putting links in the references section?" | No | The mentor's reply may involve reliable sources policy, but the mentee is not asking about rules → Q4=no |
| "How can I summarize an article?" | No | Operational question |
| "How can I fix it?" (citation error) | No | Technical question |
| "How do I get started?" | No | Direction question (Q2), not a rule question |

### No Examples — Mentions Sensitive Subject but No Norm-Seeking Language

⚠ These messages' editing subjects may objectively touch on policy (BLP/COI/notability), but the mentee's text contains no norm-seeking signals, so Q4=no.

| Message | Q4 | Explanation |
|---------|-----|------------|
| "How Can I add a page about a politician who is currently active?" | No | Although politician = sensitive subject, the mentee is only asking about operation → Q4=no |
| "I need to add an article about my late husband. He was a civil rights leader. How do I get this done?" | No | The mentee is only asking about process; no norm-seeking language. The mentor will proactively mention notability, but that is the mentor's behavior |
| "how can I create a page for my charity" | No | The mentee is only asking about operation. The mentor will mention COI, but the mentee did not ask |
| "create a page for my company" | No | Same as above |
| "I have discovered an upcoming artist and I wanna write about him" | No | The mentee is stating a goal; no norm-seeking language |
| "can I make my professor's page in Wikipedia?" | No | "can I" + no policy words → treated as operational question → Q4=no |
| "How do I find my comments on the Andrew Tate page?" | No | Operational question. Andrew Tate is a living person but the mentee is not asking about rules |
| "How can I create my own biography?" | No | "can I" + no policy words → operational question. The mentor will proactively mention COI/autobiography policy, but the mentee did not ask |
| "I was reading an article about a company I was working in and saw the logo is their old one. How to change it?" | No | Operational question (changing an image); though it involves a company, the mentee has no normative awareness |

### Borderline Cases

| Message | Q4 | Explanation |
|---------|-----|------------|
| "Why can't I contribute a new article to wikipedia?" | Yes | "can't" = being blocked → reactive norm-seeking |
| "Why is this page published on Wikipedia? This page is locked from edits too" | Yes | "locked" = enforcement result, questioning content → norm-seeking |
| "can I make a page for experience (like just a private one no one can see) I wanna practice a little first" | No | "can I" = operational question ("is there a practice area"), no policy words → Q4=no |
| "I need to take some knowledge about Wikipedia rules" | No | Mentions "rules" but has no specific rule-related question; generic expression of desire to learn → Q4=no |
| "I'll do my best to follow all the rules. I definitely don't want to be a vandal!" | No | Mentions rules and vandal but is not **asking** about rules. Q0=no (no substantive request) |
| "What to do if we have argument over an edit in wiki? How can it be resolved?" | Yes | Asking about conflict resolution procedures → norm-seeking |
| "I have secondary sources to add but when I try to add them on the Talk page, I cannot because it is semi-protected. Where can I post?" | No | Blocked by protection, but asking "where to post" (operational question), not "why am I restricted" → Q4=no |
| "I'd Be Allowed To Make My Own Wikipedia Meaning For B.B.C" | Yes | "allowed" = policy word → Q4=yes |
| "someone removed and ask for submit proof then how to upload proof" | Yes | "removed" = enforcement result → Q4=yes. Although the mentee is mainly asking about operation (how to upload), the first half describes policy enforcement |
| "I want to create a biography article of my mentor who is someone I admire" (no policy words) | No | Only mentions a sensitive subject (writing a biography for someone else) but no norm-seeking language → Q4=no. **Compare**: if the same person said "...while staying within the rules" → Q4=yes |
| "can I use grammarly to help with the editing?" | No | No policy words → treated as operational question. Although Wikipedia does have tool usage policies, the mentee is not expressing compliance concern. **Compare**: "is it **allowed** to use grammarly?" → Q4=yes |
| "Can I create two accounts?" | No | No policy words → treated as operational question. **Compare**: "Can I create two accounts, or is that **against the rules**?" → Q4=yes |
| "Can I add my own research findings to this article?" | No | No policy words → treated as operational question. **Compare**: "Can I add my own research, or does that count as **original research**?" → Q4=yes |

---

## Q5 Own Work: Has the user already done some work?

### Definition
Does the mentee come with **editing activity that has already produced observable results on Wikipedia**? There must be evidence that the editing behavior **has already occurred** — content has been saved, submitted, seen or processed by others — rather than merely being in the preparation, research, or planning stage. Q5 reflects the mentee's **current socialization stage** — whether they have already actively participated with work to point to, or are starting from scratch. Q5 does not concern whether the mentor can locate the specific content (that is Q3's concern); it only concerns whether the mentee has already produced editing behavior or results.

Distinguishing from Q3:
- **Q3**: The mentee asks the mentor to **look at and give feedback** → emphasizes the mentor's action
- **Q5**: Whether the mentee **has already done work** → emphasizes the mentee's state
- Q5=yes does not require Q3=yes. The mentee may mention past work but ask an unrelated question.

### Decision Criteria
- Has the user mentioned content they have **recently edited/created/submitted**?
- **Core judgment**: Is there evidence in the message that editing behavior **has already occurred** and is **relevant to the current request**? The key is whether there are signals of **action completed or consequences emerged**.
- ⚠ **Must be the user's own editing behavior on Wikipedia** — does not include others' activity, nor the user's real-world work outside Wikipedia (such as publishing books, writing for journals).
- **Content output signal words** (→ Q5=yes): was removed / gets messed up / keep getting error / not displaying / accidentally deleted / reverted / was declined / saved / submitted / I created / I edited / my draft / my article — these all indicate the user has already produced or modified content.
- **Account status signal words** (→ this alone is insufficient to determine Q5=yes): blocked / locked out / cannot edit / restricted / suspended — these describe account status, not content output. The user may have been blocked before producing any content.
- **Not-yet-occurred/in-preparation signal words** (→ Q5=no): want to / would like to / interested in / don't know how to / working on / planning to / preparing / researching — these indicate the user is in the intention or preparation stage and has not yet produced observable editing results on Wikipedia. "I'm working on updating X" does not equal "I updated X": the former may only involve local research and gathering materials; the latter clearly indicates the edit has been saved.
- **Distant past signal words** (→ Q5=no): years ago / a long time ago / back in [distant year] / once before. If the user only mentions editing activity from the distant past and the current request is about starting something new, the past activity is just background information and does not count as current own work.
- ⚠ Watch for phrasing traps: the first half may seem like planning, but the second half reveals it has already happened. For example, "I'm wanting to update X but it's instantly been removed" — "been removed" indicates the edit has been made and reverted → Q5=yes
- ⚠ **Article/pronoun patterns**: **my/the/this + content word** (my article / my draft / the article I wrote) usually implies existing work → Q5=yes; **a/an + content word** (an article / a draft / a page) is usually a generic question with no implication of existing work → Q5=no.
  - ⚠ **Ambiguity of "my page"/"our page"**: This may mean (a) a page the user created/edited on Wikipedia → Q5=yes, or (b) a page **about** the user or their organization → Q5=no. Distinguish by context: if the user is the subject of the page (not the editor), the possessive reference alone cannot establish editing activity. For example: "there are inaccuracies in the article about me" → user is the page subject, not the editor → Q5=no; "I edited my page yesterday" → user is the editor → Q5=yes.
  - Examples:
  - "How can I publish **an** article?" → Q5=no (generic process question)
  - "How do I get **my** article published?" → Q5=yes ("my article" = existing work)
  - "How to submit **a** draft for review?" → Q5=no (generic process question)
  - "How to submit **my** draft for review?" → Q5=yes ("my draft" = existing)
  - "How long does it take **a** page to get published?" → Q5=no (asking about general timeframe)
  - "How long does it take for **my** page to get published?" → Q5=yes ("my page" = existing)
  - "I've written an article, how do I publish it?" → Q5=yes ("I've written" = perfect tense, existing)

### Yes Examples

| Message | Q5 | Explanation |
|---------|-----|------------|
| "I wrote a draft about X [LINK]" | Yes | Has a link; the most unambiguous own work |
| "I edited the article on Y but it was reverted" | Yes | Specifies the article |
| "My article on Z was tagged for deletion" | Yes | Specifies the article |
| "I created a userbox but it's not displaying right" | Yes | Mentions a specific object they created |
| "I've made my first wiki article but the name is still saying Draft [LINK]" | Yes | Has a link |
| "I created my village wiki page called 'Jerrimadugu'" | Yes | Specifies the page name |
| "I updated a pic of Dan Barouch on Wikimedia Commons, but the article has not changed" | Yes | Specifies the article and operation |
| "I submitted my draft" (no title or link) | Yes | Though no title is given, the mentor can find it from the user's contribution history. Lean Q5=yes |
| "I'm trying to add a reference but I keep getting an error" | Yes | "keep getting an error" = has already attempted the operation and seen the result |
| "I'm wanting to update X but it's instantly been removed" | Yes | First half sounds like planning, but "been removed" = edit has occurred and been reverted |
| "I started a draft but I'm not sure if it's ready" | Yes | "started a draft" = has already created content |
| "I made some changes to an article but I'm not sure they saved" | Yes | "made some changes" = has already operated |
| "I was editing an article and accidentally deleted a section" | Yes | "was editing" + specific consequence = has occurred |
| "I added an infobox but it's not displaying correctly" | Yes | "added" + seeing result = has occurred |

### No Examples

| Message | Q5 | Explanation |
|---------|-----|------------|
| "In the article for Ron Sexsmith's 2004 album Retriever I want to add a reference" | No | Has a specific page but "I want to add" = hasn't done it yet, only planning |
| "I'm interested in editing history articles" | No | Has an interest area but no specific work |
| "What are the rules about notability?" | No | Asking about general rules; no mention of own work |
| "How do userboxes work?" | No | Asking about general operations; no mention of own work |
| "I want to create an article about X" | No | Hasn't done it yet; only planning. No existing work |
| "How do I start editing on Wikipedia?" | No | No specific work involved |
| "I'd like to write about my company" | No | Pure intention; hasn't started |
| "I'm interested in editing the article about X" | No | Interest/intention; hasn't started |
| "I want to add an infobox but don't know how" | No | Doesn't know how = hasn't started; pure question |

### Borderline Cases

| Message | Q5 | Explanation |
|---------|-----|------------|
| "My edits keep getting reverted" | Yes | "my edits" + "reverted" = mentee made edits and they were reverted → Q5=yes |
| "I've been editing X article" (no specific problem described) | Yes | Specifies the article name; mentor can go check |
| "I've been going a bit ham on editing Wikipedia, and there are a few pages [LINK], [LINK], [LINK]" | Yes | Has links; actual editing behavior |
| "this is my first wikipedia page...in the sandbox" (no link) | Yes | Mentor can find it from the user's sandbox |
| "I want to write about a particular steam launch...named 'Antic'" | No | User hasn't created a page yet; only describing a plan |

---

## Coding Process Summary

```
Step 1: Determine Q0 (substantive question/request/intent to contribute?)
  └── No → Q2-Q5=N
  └── Yes → Continue coding all

Step 2: If Q0=yes, independently determine Q2, Q3, Q4, Q5
  Q2 Referent: Lacks task direction? (yes - doesn't know what to do / no - has a clear goal)
  Q3 Appraisal: Requesting feedback on existing work? (yes/no)
  Q4 Normative: Asking about rules/policies? (yes/no)
  Q5 Own Work: Has the mentee already done work? (yes/no) — reflects socialization stage; does not concern whether the mentor can locate it
```

### Common Combination Patterns

| Combination | Typical Message | Meaning |
|-------------|----------------|---------|
| Q2-Q4=no, Q5=no | "How do I add a citation?" / "How to create an article about X?" / "I want to write a biography" | Operational question with clear goal (baseline) |
| Q2=yes | "How do I get started?" / "How to create an article?" (generic, no category) | Lacks direction; needs guidance |
| Q3=yes + Q5=yes | "Can you review my draft [LINK]?" | Seeking feedback on specific work |
| Q4=yes + Q5=yes | "Why was my edit reverted?" | Own work was rejected; seeking explanation |
| Q3=yes + Q4=yes + Q5=yes | "My draft was declined, what do I fix?" | Seeking feedback + asking rules + has specific work |

### Expected Base Rates (estimated from 500-message sample)

| Dimension | Expected Proportion | Explanation |
|-----------|-------------------|-------------|
| Q0=yes | ~80-85% | Most messages have substantive requests |
| Q2 Referent | ~10-12% | Directionless type + generic creation tasks (create article/page without category specification) |
| Q3 Appraisal | ~7-9% | Requires existing work to request evaluation |
| Q4 Normative | ~15-25% | After tightening: only when mentee text has explicit norm-seeking language (removed cases of merely mentioning sensitive subjects) |
| Q5 Own Work | ~20-25% | Completed and submitted specific pages/drafts/edits |
| Q0=yes, Q4=no, Q2=no, Q5=no | ~35-45% | Operational questions with no policy signals (baseline) |
