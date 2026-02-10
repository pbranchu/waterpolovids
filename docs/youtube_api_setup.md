# YouTube API Setup & Verification

## Problem

Unverified Google Cloud projects have a daily upload limit on the YouTube Data API. You'll see this error when the limit is hit:

```
The user has exceeded the number of videos they may upload.
```

The default quota is **10,000 units/day**. A `videos.insert` (upload) costs **1,600 units**, so you can upload ~6 videos/day with the default quota.

## Step 1: Move App Out of "Testing" Mode

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Select your project
3. Navigate to **APIs & Services > OAuth consent screen**
4. Your app is likely in **"Testing"** publishing status — this limits it to 100 test users and restricted quotas
5. Click **"Publish App"** to move to production
6. Google will show a warning about verification — for personal use with only your own account, this is fine

## Step 2: Verify the App (Optional, for Higher Quotas)

If you need more than ~6 uploads/day:

1. Go to **APIs & Services > OAuth consent screen**
2. Click **"Prepare for verification"**
3. You'll need to provide:
   - An authorized domain you own
   - A link to your app's privacy policy (can be a simple page)
   - A link to your app's homepage
4. Submit for verification — Google typically responds within **5 business days**

## Step 3: Request Quota Increase

If you need significantly more quota:

1. Fill out the [YouTube API Services - Audit and Quota Extension Form](https://support.google.com/youtube/contact/yt_api_form?hl=en)
2. Provide:
   - Your Google Cloud project number
   - A description of how you use the API
   - Why you need additional quota
3. Expect a response from the Google Compliance team within **5 business days**
4. See [Quota and Compliance Audits](https://developers.google.com/youtube/v3/guides/quota_and_compliance_audits) for details

## Quick Fix: Wait for Quota Reset

The daily quota resets at **midnight Pacific Time**. If you just need to upload a few more videos, wait until tomorrow.

## Checking Current Quota Usage

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Navigate to **APIs & Services > Dashboard**
3. Click on **YouTube Data API v3**
4. Check the **Quotas** tab to see current usage vs. limit

## References

- [YouTube Data API Quota Calculator](https://developers.google.com/youtube/v3/determine_quota_cost)
- [YouTube API Quota and Compliance Audits](https://developers.google.com/youtube/v3/guides/quota_and_compliance_audits)
- [YouTube API Audit and Quota Extension Form](https://support.google.com/youtube/contact/yt_api_form?hl=en)
- [OAuth Application Rate Limits](https://support.google.com/cloud/answer/9028764?hl=en)
