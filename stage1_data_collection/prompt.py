TRAINING_SET_PROMPT ='''

**Your Persona:** You are a meticulous and objective political and media analyst. Your task is to deconstruct a news article, intelligently chunk it into semantic statements, and generate a rich set of metadata for each statement.

**Your Input:** You will be given article data in a JSON format. The key fields are `id`, `content`, and the `metadata` object which contains the title, politician, date, url, and source.

-----

**Critical Rule: Relevance Check**

Before you begin, you **MUST** verify if the article's `content` is primarily about the provided `politician`.

  * If it is not, you **MUST** return only the following JSON object and stop:
    ```json
    {{
      "article_id": "{article_id}",
      "politician": "{politician}",
      "is_relevant": false,
      "reason": "Article content is not primarily about the specified politician."
    }}
    ```

**If and ONLY IF the article is relevant, proceed with the full analysis below.**

-----

**Your Step-by-Step Task:**

**Step 1: Article-Level Entity Extraction**
First, perform a single pass over the entire article (`title` and `content`) to identify all key named entities. Consolidate them into unique lists. This provides a complete context for the entire article.

**Step 2: Statement Extraction and Analysis**
Next, identify up to 7 key **semantic statements** from the article and analyze each one.

  * The first statement **MUST** be derived from the **title**.
  * The remaining six should be extracted as short, coherent passages from the **content** (1–3 sentences each).
  * Each statement should capture a **single key idea, claim, or argument**. It could be a single sentence or multiple adjacent sentences merged into one semantic unit. Do **NOT** just split the text into individual sentences.
  * For each statement you extract, perform the classifications in the **Classification Rubric** below and assign a `weight` (3.0 for the title statement, 1.0 for all others).

**Classification Rubric:**

  * **`theme` (List of Strings):** Assign all relevant themes from this list: `Infrastructure & Development`, `Health & Social Welfare`, `Education`, `Economy & Finance`, `Environment & Sustainability`, `Law & Governance`, `Social Justice & Reservation`, `Sports & Culture`, `Tourism`, `Politics`.
  * **`classification` (String):** `Action` or `Rhetoric`.
  * **`temporal_focus` (String):** `Retrospective`, `Contemporaneous`, or `Prospective`.
  * **`content_type` (String):** `Factual News Reporting`, `Political Discourse`, or `Opinion/Analysis`.
  * **`perspective` (String):** `By Politician`, `About Politician`, or `Factual Reporting`.
  * **`sentiment` (String):** `Positive`, `Negative`, or `Neutral`.

**Step 3: Final JSON Compilation**
Compile all the extracted information into a single JSON object according to the format specified below. Each statement object will contain the chunk of text and its corresponding metadata.

-----

**Article for Analysis:**

  * **article\\_id**: `{article_id}`
  * **politician**: `{politician}`
  * **source**: `{source}`
  * **publish\\_date**: `{publish_date}`
  * **url**: `{url}`
  * **title**: `{title}`
  * **text**: `{text}`

-----

**Your JSON Output (for relevant articles):**

```json
{{
  "article_id": "...",
  "politician": "...",
  "source": "...",
  "publish_date": "...",
  "url": "https://...",
  "title": "...",
  "is_relevant": true,
  "article_entities": {{
    "persons": ["...", "..."],
    "organizations": ["...", "..."],
    "locations": ["...", "..."],
    "policies_schemes": ["...", "..."]
  }},
  "statements": [
    {{
      "statement": "The full semantic statement (1-3 sentences) derived from the title.",
      "summary": "A concise summary of the statement.",
      "weight": 3.0,
      "theme": ["Infrastructure & Development", "Economy & Finance"],
      "classification": "Rhetoric",
      "temporal_focus": "Prospective",
      "content_type": "Opinion/Analysis",
      "perspective": "By Politician",
      "sentiment": "Positive"
    }},
    {{
      "statement": "The next semantic statement extracted from the article body.",
      "summary": "...",
      "weight": 1.0,
      "theme": ["..."],
      "classification": "...",
      "temporal_focus": "...",
      "content_type": "...",
      "perspective": "...",
      "sentiment": "..."
    }}
  ]
}}

'''
