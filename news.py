import os
import re
import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from time import sleep
from dotenv import load_dotenv
from groq import Groq
import feedparser #type:ignore
import pandas as pd #type:ignore
from datetime import datetime


load_dotenv()

my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("invalid api key")

client = Groq(api_key=my_api_key)

model = "openai/gpt-oss-120b"


user_interests = """
hey i am a computer science graduate and im interested in topics like
geopolitics, banking, finance, gaming, technology, science
"""


def ask_llm(system_prompt, user_prompt):

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )

    return response.choices[0].message.content


def step1():

    system_prompt = """
You are a professional user-interest extraction assistant.

Your task is to extract the topics and interests explicitly mentioned
by the user.

Rules:
- Return only the interests.
- Do not invent or infer interests.
- Do not include personal information such as education or occupation
  unless it is explicitly presented as an interest.
- Keep each interest short and clear.
- Remove duplicate interests.
- Normalize capitalization.
- Return the result as a comma-separated list.
- If no interests are found, return "NONE".
"""

    user_prompt = f"""
Extract the user's interests from the following text:

{user_interests}
"""

    result = ask_llm(system_prompt, user_prompt)

    interests = [
        interest.strip().lower()
        for interest in result.split(",")
        if interest.strip()
    ]

    return interests


def news_collection():

    RSS_URL = [
        "http://feeds.bbci.co.uk/news/rss.xml",
        "http://feeds.bbci.co.uk/news/technology/rss.xml",
        "http://feeds.bbci.co.uk/news/business/rss.xml"
    ]

    articles = []

    for rss_url in RSS_URL:

        feed = feedparser.parse(rss_url)

        for item in feed.entries:

            title = item.get("title", "").strip()
            timestamp = item.get("published", "").strip()
            link = item.get("link", "").strip()
            summary = item.get("summary", "").strip()

            if not title:
                continue

            articles.append({
                "title": title,
                "timestamp": timestamp,
                "summary": summary,
                "link": link,
                "source": "BBC"
            })

    df = pd.DataFrame(articles)

    df = df.drop_duplicates(subset=["title"])

    df = df.dropna(subset=["title"])

    df["scraped_at"] = datetime.now().isoformat()

    df.to_csv("news.csv", index=False)

    print(f"Saved {len(df)} articles")

    return df


def clean_and_filter(df, interests):

    df["title"] = (
        df["title"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["summary"] = (
        df["summary"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["link"] = (
        df["link"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df = df[df["title"] != ""]

    df = df.drop_duplicates(subset=["title"])

    df["search_text"] = (
        df["title"] + " " + df["summary"]
    ).str.lower()

    interest_keywords = {

        "geopolitics": [
            "geopolitics",
            "war",
            "conflict",
            "ukraine",
            "russia",
            "nato",
            "china",
            "taiwan",
            "middle east",
            "israel",
            "iran",
            "trump",
            "election",
            "diplomacy"
        ],

        "banking": [
            "bank",
            "banks",
            "banking",
            "central bank",
            "federal reserve",
            "fed",
            "interest rate",
            "lending",
            "loan",
            "deposit"
        ],

        "finance": [
            "finance",
            "financial",
            "stock",
            "stocks",
            "market",
            "markets",
            "shares",
            "investment",
            "investor",
            "inflation",
            "economy",
            "economic",
            "wall street",
            "ipo",
            "crypto"
        ],

        "gaming": [
            "gaming",
            "game",
            "games",
            "video game",
            "playstation",
            "xbox",
            "nintendo",
            "steam",
            "esports"
        ],

        "technology": [
            "technology",
            "tech",
            "artificial intelligence",
            "ai",
            "software",
            "robot",
            "robotics",
            "computer",
            "digital",
            "tesla"
        ],

        "science": [
            "science",
            "scientist",
            "research",
            "climate",
            "environment",
            "space",
            "physics",
            "biology",
            "chemistry"
        ]
    }

    filtered_articles = []

    for _, article in df.iterrows():

        text = article["search_text"]

        matched_interests = []

        for interest in interests:

            interest = interest.lower().strip()

            keywords = interest_keywords.get(
                interest,
                [interest]
            )

            if any(keyword in text for keyword in keywords):

                matched_interests.append(interest)

        if matched_interests:

            article["matched_interests"] = ", ".join(
                matched_interests
            )

            filtered_articles.append(article)

    filtered_df = pd.DataFrame(filtered_articles)

    if not filtered_df.empty:

        filtered_df = filtered_df.drop(
            columns=["search_text"]
        )

    print(
        f"Total articles collected: {len(df)}"
    )

    print(
        f"Articles after Python pre-filter: {len(filtered_df)}"
    )

    return filtered_df


def step2(filtered_news, interests):

    if filtered_news.empty:

        print(
            "No articles available for LLM filtering."
        )

        return []

    news_data = filtered_news[
        [
            "title",
            "timestamp",
            "summary",
            "link",
            "source",
            "matched_interests"
        ]
    ].head(10).to_dict(orient="records")

    system_prompt = """
You are a professional news relevance analyst.

Evaluate every article based on the user's interests.

For every article:
- Give a relevance score from 0 to 100.
- Identify the strongest matching user interest.
- Give a short reason.
- Do not invent information.
- Use only the provided article information.
- Do not modify the title.
- Do not modify the link.
- Do not modify the source.
- Preserve the article summary.

Scoring:

90-100 = Extremely relevant
80-89 = Highly relevant
70-79 = Clearly relevant
50-69 = Somewhat relevant
30-49 = Weakly relevant
0-29 = Not relevant

Return ONLY valid JSON.

Return exactly this structure:

{
    "articles": [
        {
            "title": "original title",
            "timestamp": "original timestamp",
            "summary": "original summary",
            "link": "original link",
            "source": "BBC",
            "matched_interest": "finance",
            "relevance_score": 85,
            "reason": "Short explanation"
        }
    ]
}
"""

    user_prompt = f"""
USER INTERESTS:

{json.dumps(interests, ensure_ascii=False)}

NEWS ARTICLES:

{json.dumps(news_data, ensure_ascii=False)}
"""

    result = ask_llm(
        system_prompt,
        user_prompt
    )

    result = result.strip()

    result = re.sub(
        r"^```(?:json)?\s*",
        "",
        result,
        flags=re.IGNORECASE
    )

    result = re.sub(
        r"\s*```$",
        "",
        result
    )

    start = result.find("{")
    end = result.rfind("}")

    if start == -1 or end == -1:

        print("LLM did not return valid JSON.")
        print(result)

        return []

    result = result[start:end + 1]

    try:

        parsed = json.loads(result)

    except json.JSONDecodeError:

        print("LLM did not return valid JSON.")
        print(result)

        return []

    if isinstance(parsed, dict):

        results = parsed.get(
            "articles",
            []
        )

    elif isinstance(parsed, list):

        results = parsed

    else:

        return []

    relevant_articles = []

    for article in results:

        if not isinstance(article, dict):
            continue

        score = article.get(
            "relevance_score",
            0
        )

        try:

            score = int(score)

        except:

            score = 0

        if score >= 70:

            relevant_articles.append({

                "title": article.get(
                    "title",
                    ""
                ),

                "timestamp": article.get(
                    "timestamp",
                    ""
                ),

                "summary": article.get(
                    "summary",
                    ""
                ),

                "link": article.get(
                    "link",
                    ""
                ),

                "source": article.get(
                    "source",
                    "BBC"
                ),

                "matched_interest": article.get(
                    "matched_interest",
                    ""
                ),

                "relevance_score": score,

                "reason": article.get(
                    "reason",
                    ""
                )
            })

    print(
        f"Articles evaluated by LLM: {len(results)}"
    )

    print(
        f"Articles with score >= 70: {len(relevant_articles)}"
    )

    return relevant_articles


def step3(relevant_news, interests):

    if not relevant_news:

        return []

    news_data = []

    for article in relevant_news:

        news_data.append({

            "title": article.get(
                "title",
                ""
            ),

            "summary": article.get(
                "summary",
                ""
            ),

            "interest": article.get(
                "matched_interest",
                ""
            ),

            "source": article.get(
                "source",
                ""
            ),

            "link": article.get(
                "link",
                ""
            ),

            "relevance_score": article.get(
                "relevance_score",
                0
            ),

            "reason": article.get(
                "reason",
                ""
            )
        })

    system_prompt = """
You are a professional news summarization assistant.

Summarize the provided news articles for a personalized daily news briefing.

Rules:
- Return ONLY valid JSON.
- Do not use Markdown.
- Do not use ```json or ``` .
- Return a JSON object containing an "articles" array.
- Keep exactly these fields for every article:
  title
  summary
  why_it_matters
  interest
  source
  link
  relevance_score
- Do not invent facts.
- Do not change the title.
- Keep each summary between 2 and 3 sentences.
- Keep why_it_matters to 1 sentence.
- Preserve the original source and link.
- Preserve the relevance score.
"""

    user_prompt = f"""
USER INTERESTS:

{json.dumps(interests, ensure_ascii=False)}

ARTICLES:

{json.dumps(news_data, ensure_ascii=False)}
"""

    result = ask_llm(
        system_prompt,
        user_prompt
    )

    result = result.strip()

    result = re.sub(
        r"^```(?:json)?\s*",
        "",
        result,
        flags=re.IGNORECASE
    )

    result = re.sub(
        r"\s*```$",
        "",
        result
    )

    start = result.find("{")
    end = result.rfind("}")

    if start == -1 or end == -1:

        print("Step 3 did not return valid JSON.")
        print(result)

        return []

    result = result[start:end + 1]

    try:

        parsed = json.loads(result)

    except json.JSONDecodeError:

        print("Step 3 returned invalid JSON.")
        print(result)

        return []

    if isinstance(parsed, dict):

        summarized_news = parsed.get(
            "articles",
            []
        )

    elif isinstance(parsed, list):

        summarized_news = parsed

    else:

        return []

    cleaned_news = []

    for article in summarized_news:

        if not isinstance(article, dict):

            continue

        cleaned_news.append({

            "title": article.get(
                "title",
                ""
            ),

            "summary": article.get(
                "summary",
                ""
            ),

            "why_it_matters": article.get(
                "why_it_matters",
                ""
            ),

            "interest": article.get(
                "interest",
                ""
            ),

            "source": article.get(
                "source",
                "BBC"
            ),

            "link": article.get(
                "link",
                ""
            ),

            "relevance_score": article.get(
                "relevance_score",
                0
            )
        })

    return cleaned_news


def send_email(summarized_news):

    sender_email = os.getenv(
        "GMAIL_ADDRESS"
    )

    sender_password = os.getenv(
        "GMAIL_APP_PASSWORD"
    )

    receiver_email = os.getenv(
        "EMAIL_TO"
    )

    if not sender_email or not sender_password or not receiver_email:

        raise ValueError(
            "Email credentials are missing from .env"
        )

    message = MIMEMultipart(
        "alternative"
    )

    message["Subject"] = (
        "Your Personalized News Briefing"
    )

    message["From"] = sender_email

    message["To"] = receiver_email

    html = """
<html>
<body>

<h2>Personalized News Briefing</h2>
"""

    for article in summarized_news:

        title = article.get(
            "title",
            "Untitled"
        )

        summary = article.get(
            "summary",
            ""
        )

        why_it_matters = article.get(
            "why_it_matters",
            ""
        )

        interest = article.get(
            "interest",
            "General"
        )

        source = article.get(
            "source",
            "Unknown"
        )

        link = article.get(
            "link",
            "#"
        )

        relevance_score = article.get(
            "relevance_score",
            0
        )

        html += f"""
<hr>

<h3>{title}</h3>

<p>
<b>Category:</b> {interest}
</p>

<p>
<b>Relevance:</b> {relevance_score}/100
</p>

<p>
<b>Summary:</b><br>
{summary}
</p>

<p>
<b>Why it matters:</b><br>
{why_it_matters}
</p>

<p>
<b>Source:</b> {source}
</p>

<p>
<a href="{link}">
Read full article
</a>
</p>
"""

    html += """
<hr>

<p>
This briefing was generated automatically.
</p>

</body>
</html>
"""

    message.attach(
        MIMEText(
            html,
            "html"
        )
    )

    with smtplib.SMTP(
        "smtp.gmail.com",
        587
    ) as server:

        server.starttls()

        server.login(
            sender_email,
            sender_password
        )

        server.sendmail(
            sender_email,
            receiver_email,
            message.as_string()
        )

    print(
        "Email sent successfully."
    )


def main():

    interests = step1()

    print(
        "\nUSER INTERESTS:"
    )

    print(interests)

    sleep(2)

    news_df = news_collection()

    filtered_news = clean_and_filter(
        news_df,
        interests
    )

    sleep(2)

    relevant_news = step2(
        filtered_news,
        interests
    )

    sleep(2)

    summarized_news = step3(
        relevant_news,
        interests
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "PERSONALIZED NEWS BRIEFING"
    )

    print(
        "=" * 70
    )

    if not summarized_news:

        print(
            "No relevant news found."
        )

        return

    print(
        json.dumps(
            summarized_news,
            indent=4,
            ensure_ascii=False
        )
    )

    send_email(
        summarized_news
    )


if __name__ == "__main__":

    main()