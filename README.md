# Personalized AI News Briefing

An AI-powered personalized news recommendation system that collects news articles from RSS feeds, filters them based on user interests, uses an LLM to evaluate relevance, generates personalized summaries, and delivers the final news briefing through email.

## Features

- Collects news articles from BBC RSS feeds
- Stores collected articles locally
- Filters articles using user-defined interests
- Uses an LLM to evaluate article relevance
- Assigns a relevance score to each article
- Generates personalized summaries
- Explains why each article matters to the user
- Selects highly relevant articles automatically
- Sends the final personalized news briefing through email

## User Interests

The current system is configured for:

- Geopolitics
- Banking
- Finance
- Gaming
- Technology
- Science

## How It Works

```text
BBC RSS Feeds
      ↓
Article Collection
      ↓
Local Storage
      ↓
Python Pre-filtering
      ↓
LLM Relevance Evaluation
      ↓
Relevance Score
      ↓
Highly Relevant Articles
      ↓
AI-generated Summary
      ↓
Personalized News Briefing
      ↓
Email Delivery
