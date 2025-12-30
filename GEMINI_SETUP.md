# Gemini API Integration - Setup Instructions

## What Changed

The chatbot now uses **Google's Gemini API** instead of local Hugging Face models. This provides:

- ✅ Better answer quality (Gemini is a large, production-grade model)
- ✅ Faster responses (no local model loading)
- ✅ Lower RAM usage (no need to keep 1GB models in memory)
- ✅ Works on any machine (no GPU or high-end CPU needed)

## Security: API Key Protection

Your Gemini API key is stored in `.env` file which is:

- ✅ Already in `.gitignore` (won't be pushed to GitHub)
- ✅ Only loaded locally on your machine
- ⚠️ **Never commit .env to GitHub!**

## Setup Instructions

### 1. Install New Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Your .env File

The `.env` file should already exist with your API key:

```env
GEMINI_API_KEY=your-api-key-here
GEMINI_MODEL=gemini-pro
LLA_TEMPERATURE=0.6
LLA_MAX_NEW_TOKENS=512
```

### 3. Run Locally

```bash
streamlit run app.py
```

## Deployment to Streamlit Cloud

When deploying to Streamlit Cloud:

1. **Don't push .env to GitHub** (it's already ignored)
2. **Set secrets in Streamlit Cloud dashboard:**
   - Go to: https://share.streamlit.io/
   - Select your app → Settings → Secrets
   - Add this:
     ```toml
     GEMINI_API_KEY = "your-api-key-here"
     GEMINI_MODEL = "gemini-pro"
     ```

## Model Options

You can change the Gemini model in `.env`:

- **gemini-pro** (default, recommended) - Stable, reliable, good quality
- **gemini-1.5-pro** - Newer model with enhanced capabilities
- **gemini-1.5-flash** - Fast but may not be available in all API versions

## Cost Estimate

Gemini API pricing:

- **gemini-1.5-flash**: Free tier = 15 requests/min, 1500 requests/day
- After free tier: $0.075 per 1M input tokens, $0.30 per 1M output tokens

For 100 queries/day with 2K tokens/query:

- **Cost: ~$0.05/day** (almost free!)

## Testing

Test with a simple query:

```
Question: "What is an FIR?"
```

Expected: Should return India-specific answer about First Information Reports.

## Troubleshooting

**Error: "GEMINI_API_KEY environment variable not set"**

- Solution: Make sure `.env` file exists and contains `GEMINI_API_KEY=...`

**Error: "Invalid API key"**

- Solution: Verify your API key at https://makersuite.google.com/app/apikey

**Slow responses:**

- Try using `gemini-1.5-flash` instead of `gemini-1.5-pro` (faster)

## Rolling Back to Local Models

If you want to revert to local Hugging Face models, run:

```bash
git log --oneline  # Find the commit before Gemini integration
git revert <commit-hash>
```
