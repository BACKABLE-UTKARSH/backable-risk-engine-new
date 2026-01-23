# Risk Engine - Environment Setup Guide

## Overview
All sensitive credentials (API keys, database passwords, connection strings) are now stored in environment variables instead of hardcoded in the source code.

## Local Development Setup

### 1. Create `.env` file
Copy the example file and fill in your actual values:
```bash
cp .env.example .env
```

### 2. Edit `.env` file
Open `.env` and replace all placeholder values with your actual credentials:

```env
# Database
DB_PASSWORD=your_actual_password_here

# Azure Storage
AZURE_STORAGE_CONNECTION_STRING=your_actual_connection_string

# Gemini API Keys (comma-separated)
GEMINI_API_KEYS_RISK=key1,key2,key3,...

# User Database
USER_DB_PASSWORD=your_user_db_password
```

### 3. Vertex AI Credentials (Optional)

**Option A: File-based (Local Development)**
- Get `vertex-key.json` from Google Cloud Console
- Place it in the deployment directory
- The code will automatically detect and use it

**Option B: Environment Variable (Production)**
- Copy the entire JSON content of your service account key
- Set it as `GOOGLE_APPLICATION_CREDENTIALS_JSON` in your `.env`

### 4. Install Dependencies
```bash
pip install -r requirements.txt
pip install python-dotenv  # If not in requirements.txt
```

### 5. Run Locally
```bash
python "BACKABLE NEW INFRASTRUCTURE RISK ENGINE.py"
```

## Azure Deployment Setup

### 1. Set Environment Variables in Azure
Go to Azure Portal → Your App Service → Configuration → Application Settings

Add the following environment variables:

| Name | Value | Example |
|------|-------|---------|
| `DB_HOST` | Database host | `memberchat-db.postgres.database.azure.com` |
| `DB_USER` | Database username | `backable` |
| `DB_PASSWORD` | Database password | `your_password` |
| `DB_PORT` | Database port | `5432` |
| `USER_DB_HOST` | User DB host | `philotimo-staging-db.postgres.database.azure.com` |
| `USER_DB_NAME` | User DB name | `philotimodb` |
| `USER_DB_USER` | User DB username | `wchen` |
| `USER_DB_PASSWORD` | User DB password | `your_password` |
| `AZURE_STORAGE_CONNECTION_STRING` | Full connection string | `DefaultEndpoints...` |
| `GEMINI_API_KEYS_RISK` | Comma-separated API keys | `key1,key2,key3` |
| `GOOGLE_APPLICATION_CREDENTIALS_JSON` | Vertex AI service account JSON | `{"type":"service_account"...}` |

### 2. Save and Restart
- Click "Save" in Azure Portal
- Restart the App Service

## Environment Variables Reference

### Required Variables

#### Database Configuration
- `DB_HOST` - PostgreSQL host for Backable databases
- `DB_USER` - PostgreSQL username
- `DB_PASSWORD` - PostgreSQL password ⚠️ **REQUIRED**
- `DB_PORT` - PostgreSQL port (default: 5432)

#### User Database (Separate)
- `USER_DB_HOST` - User database host
- `USER_DB_NAME` - User database name
- `USER_DB_USER` - User database username
- `USER_DB_PASSWORD` - User database password ⚠️ **REQUIRED**
- `USER_DB_PORT` - User database port (default: 5432)

#### Azure Storage
- `AZURE_STORAGE_CONNECTION_STRING` - Full Azure Storage connection string ⚠️ **REQUIRED**

#### Gemini API Keys
- `GEMINI_API_KEYS_RISK` - Comma-separated list of Gemini API keys ⚠️ **REQUIRED**
  - Format: `key1,key2,key3,key4,key5`
  - Minimum: 1 key recommended
  - Optimal: 10 keys for load balancing

### Optional Variables

#### Vertex AI (Primary AI Method)
- `GOOGLE_APPLICATION_CREDENTIALS_JSON` - Service account JSON (paste entire JSON)
  - Alternative: Place `vertex-key.json` in deployment directory

#### Application Settings
- `PORT` - API server port (default: 8001)

## Security Best Practices

### ✅ DO
- Store `.env` file locally only
- Add `.env` to `.gitignore`
- Use Azure Key Vault for production secrets (advanced)
- Rotate API keys regularly
- Use different credentials for dev/staging/production

### ❌ DON'T
- Commit `.env` files to Git
- Share `.env` files via email/Slack
- Hardcode credentials in source code
- Use production credentials in development
- Commit `vertex-key.json` to Git

## Troubleshooting

### Error: "No Gemini API keys found"
**Solution:** Check that `GEMINI_API_KEYS_RISK` is set and formatted correctly (comma-separated, no spaces)

### Error: "Database connection failed"
**Solution:** Verify `DB_PASSWORD` is set correctly

### Error: "Vertex AI not available"
**Solution:**
- Check `GOOGLE_APPLICATION_CREDENTIALS_JSON` is set (Azure)
- Or ensure `vertex-key.json` exists in directory (local)
- Verify JSON format is valid

### Error: "Blob upload failed"
**Solution:** Verify `AZURE_STORAGE_CONNECTION_STRING` is complete and correct

## Migration from Hardcoded Credentials

If you have an old version with hardcoded credentials:

1. Extract all hardcoded values
2. Add them to your `.env` file
3. Deploy the new version
4. Verify everything works
5. Delete any files with old hardcoded credentials

## Support

For issues or questions:
- Check logs for specific error messages
- Verify all **REQUIRED** environment variables are set
- Test with a single API key first before adding multiple keys
