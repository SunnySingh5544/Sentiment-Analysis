from flask import Flask, request, jsonify
from flask_cors import CORS
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv
import pickle
import re
import os
import traceback

# ── Load .env ──
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")

if not API_KEY:
    print("ERROR: YOUTUBE_API_KEY not found in .env file")
    # exit(1)
else:
    print(f"✓ API Key loaded")

app = Flask(__name__)
CORS(app)

# ── Load model ──
print("Loading trained model...")
try:
    with open("sentiment_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    print("✓ Model loaded")
except FileNotFoundError as e:
    print(f"ERROR: {e}")
    exit(1)


# ────────────────────────────────────────────
# URL PARSING
# ────────────────────────────────────────────

def extract_video_id(url):
    """Extract 11-char video ID from any YouTube URL."""
    patterns = [
        r'(?:v=)([0-9A-Za-z_-]{11})',
        r'youtu\.be\/([0-9A-Za-z_-]{11})',
        r'embed\/([0-9A-Za-z_-]{11})',
        r'shorts\/([0-9A-Za-z_-]{11})',
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


def extract_playlist_id(url):
    """Extract playlist ID from list= parameter."""
    m = re.search(r'[?&]list=([A-Za-z0-9_-]+)', url)
    return m.group(1) if m else None


def detect_url_type(url):
    """
    Returns ('video', id), ('playlist', id) or (None, None).
    If URL has BOTH v= and list=, we treat it as a single video
    (user is watching one video that happens to be in a playlist).
    To analyse the full playlist they should use the playlist URL.
    """
    pid = extract_playlist_id(url)
    vid = extract_video_id(url)

    if vid and pid:
        # has both — treat as single video
        return 'video', vid
    if pid:
        return 'playlist', pid
    if vid:
        return 'video', vid
    return None, None


# ────────────────────────────────────────────
# TEXT / MODEL HELPERS
# ────────────────────────────────────────────

def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.lower().strip()


def predict_sentiment(text):
    cleaned = clean_text(text)
    if len(cleaned) < 3:
        return "neutral", 0.0
    vec   = vectorizer.transform([cleaned])
    pred  = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    conf  = round(float(max(proba)), 3)
    return ("positive", conf) if pred == 1 else ("negative", round(-conf, 3))


def analyse_comments(raw_comments):
    """Run model on a list of {raw, likes} dicts."""
    result = []
    for c in raw_comments:
        label, score = predict_sentiment(c["raw"])
        if abs(score) < 0.55:
            label = "neutral"
        result.append({
            "text":      c["raw"][:250],
            "sentiment": label,
            "score":     score,
            "likes":     c["likes"],
        })
    return result


def build_stats(analysed):
    """Compute aggregate stats from analysed comment list."""
    total = len(analysed)
    if total == 0:
        return None

    pos = [c for c in analysed if c["sentiment"] == "positive"]
    neg = [c for c in analysed if c["sentiment"] == "negative"]
    neu = [c for c in analysed if c["sentiment"] == "neutral"]

    pos_pct = round(len(pos) / total * 100, 1)
    neg_pct = round(len(neg) / total * 100, 1)
    neu_pct = round(len(neu) / total * 100, 1)
    avg     = round(sum(c["score"] for c in analysed) / total, 3)

    if pos_pct >= 60:       overall = "Mostly Positive"
    elif neg_pct >= 60:     overall = "Mostly Negative"
    elif pos_pct > neg_pct: overall = "Slightly Positive"
    elif neg_pct > pos_pct: overall = "Slightly Negative"
    else:                   overall = "Mixed / Neutral"

    return {
        "total":        total,
        "positive":     len(pos),
        "negative":     len(neg),
        "neutral":      len(neu),
        "pos_pct":      pos_pct,
        "neg_pct":      neg_pct,
        "neu_pct":      neu_pct,
        "avg_score":    avg,
        "overall":      overall,
        "top_positive": sorted(pos, key=lambda x: x["likes"], reverse=True)[:3],
        "top_negative": sorted(neg, key=lambda x: x["likes"], reverse=True)[:3],
        "all_comments": analysed[:60],
    }


# ────────────────────────────────────────────
# YOUTUBE API HELPERS
# ────────────────────────────────────────────

def build_youtube():
    return build("youtube", "v3", developerKey=API_KEY)


def get_video_info(youtube, video_id):
    """Return snippet + statistics for a single video."""
    res = youtube.videos().list(
        part="snippet,statistics",
        id=video_id
    ).execute()
    if not res.get("items"):
        return None
    item    = res["items"][0]
    snippet = item["snippet"]
    stats   = item["statistics"]
    thumbs  = snippet.get("thumbnails", {})
    thumb   = (thumbs.get("high") or thumbs.get("medium") or thumbs.get("default") or {}).get("url", "")
    return {
        "title":     snippet.get("title", ""),
        "channel":   snippet.get("channelTitle", ""),
        "thumbnail": thumb,
        "views":     int(stats.get("viewCount", 0)),
        "likes":     int(stats.get("likeCount", 0)),
    }


def get_playlist_info(youtube, playlist_id):
    """Return basic info about a playlist."""
    try:
        res = youtube.playlists().list(
            part="snippet,contentDetails",
            id=playlist_id
        ).execute()
        print(f"  playlist info response: {res}")

        if not res.get("items"):
            return None

        item    = res["items"][0]
        snippet = item["snippet"]
        thumbs  = snippet.get("thumbnails", {})
        thumb   = (thumbs.get("high") or thumbs.get("medium") or thumbs.get("default") or {}).get("url", "")
        total   = item.get("contentDetails", {}).get("itemCount", "?")

        return {
            "title":       snippet.get("title", "Unknown Playlist"),
            "channel":     snippet.get("channelTitle", ""),
            "thumbnail":   thumb,
            "video_count": total,
        }
    except HttpError as e:
        print(f"  HttpError fetching playlist info: {e}")
        raise


def get_playlist_video_ids(youtube, playlist_id, max_videos=10):
    """
    Fetch up to max_videos video IDs from a playlist.
    Handles pagination and skips deleted/private videos.
    """
    video_ids = []
    next_page = None

    print(f"  Fetching video IDs for playlist: {playlist_id}")

    while len(video_ids) < max_videos:
        try:
            req = youtube.playlistItems().list(
                part="contentDetails,status",
                playlistId=playlist_id,
                maxResults=min(50, max_videos - len(video_ids)),
                pageToken=next_page,
            )
            res = req.execute()
        except HttpError as e:
            print(f"  HttpError fetching playlist items: {e}")
            raise

        items = res.get("items", [])
        print(f"  Page returned {len(items)} items")

        for item in items:
            vid_id     = item["contentDetails"].get("videoId")
            privacy    = item.get("status", {}).get("privacyStatus", "public")
            if vid_id and privacy == "public":
                video_ids.append(vid_id)

        next_page = res.get("nextPageToken")
        if not next_page or len(video_ids) >= max_videos:
            break

    print(f"  Total public video IDs found: {len(video_ids)}")
    return video_ids


def collect_comments(youtube, video_id, max_pages=2):
    """
    Collect top-level comments for a video.
    Returns list of {raw, likes}.
    Silently skips videos with disabled comments.
    """
    comments  = []
    next_page = None

    for page_num in range(max_pages):
        try:
            req = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=next_page,
                order="relevance",
                textFormat="plainText",
            )
            res = req.execute()
        except HttpError as e:
            reason = str(e)
            if "commentsDisabled" in reason or "403" in reason or "404" in reason:
                print(f"    Comments disabled/unavailable for {video_id}")
            else:
                print(f"    HttpError on page {page_num} for {video_id}: {e}")
            break

        for item in res.get("items", []):
            s = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "raw":   s.get("textDisplay", ""),
                "likes": int(s.get("likeCount", 0)),
            })

        next_page = res.get("nextPageToken")
        if not next_page:
            break

    return comments


# ────────────────────────────────────────────
# ROUTES
# ────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "running", "api_key_loaded": bool(API_KEY)})


@app.route('/preview', methods=['POST'])
def preview():
    data     = request.json or {}
    url      = data.get('url', '').strip()
    url_type, entity_id = detect_url_type(url)

    print(f"\n[PREVIEW] url_type={url_type}  entity_id={entity_id}  url={url}")

    if not entity_id:
        return jsonify({"error": "Invalid YouTube URL. Make sure it contains a valid video or playlist link."}), 400

    try:
        youtube = build_youtube()

        if url_type == 'playlist':
            info = get_playlist_info(youtube, entity_id)
            if not info:
                return jsonify({"error": "Playlist not found or is private."}), 404
            return jsonify({
                "type":        "playlist",
                "playlist_id": entity_id,
                "title":       info["title"],
                "channel":     info["channel"],
                "thumbnail":   info["thumbnail"],
                "video_count": info["video_count"],
            })

        else:
            info = get_video_info(youtube, entity_id)
            if not info:
                return jsonify({"error": "Video not found or is private."}), 404
            return jsonify({
                "type":      "video",
                "video_id":  entity_id,
                "title":     info["title"],
                "channel":   info["channel"],
                "thumbnail": info["thumbnail"],
                "views":     info["views"],
                "likes":     info["likes"],
            })

    except HttpError as e:
        msg = f"YouTube API error: {e.reason if hasattr(e,'reason') else str(e)}"
        print(f"  [PREVIEW ERROR] {msg}")
        return jsonify({"error": msg}), 500
    except Exception as e:
        print(f"  [PREVIEW ERROR] {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


@app.route('/analyse', methods=['POST'])
def analyse():
    data     = request.json or {}
    url      = data.get('url', '').strip()
    url_type, entity_id = detect_url_type(url)

    print(f"\n[ANALYSE] url_type={url_type}  entity_id={entity_id}  url={url}")

    if not entity_id:
        return jsonify({"error": "Invalid YouTube URL."}), 400

    try:
        youtube = build_youtube()

        # ── PLAYLIST ──
        if url_type == 'playlist':
            info = get_playlist_info(youtube, entity_id)
            if not info:
                return jsonify({"error": "Playlist not found or is private."}), 404

            video_ids = get_playlist_video_ids(youtube, entity_id, max_videos=10)
            if not video_ids:
                return jsonify({"error": "No public videos found in this playlist."}), 400

            # Batch-fetch video titles
            vid_details_res = youtube.videos().list(
                part="snippet",
                id=",".join(video_ids)
            ).execute()
            vid_titles = {
                item["id"]: item["snippet"]["title"]
                for item in vid_details_res.get("items", [])
            }

            # Collect + analyse per video
            all_raw   = []
            per_video = []

            for vid_id in video_ids:
                print(f"  Processing video: {vid_id}")
                raw          = collect_comments(youtube, vid_id, max_pages=2)
                analysed_vid = analyse_comments(raw)
                vid_stats    = build_stats(analysed_vid) if analysed_vid else None

                per_video.append({
                    "video_id":      vid_id,
                    "title":         vid_titles.get(vid_id, vid_id),
                    "url":           f"https://youtube.com/watch?v={vid_id}",
                    "comment_count": len(analysed_vid),
                    "stats":         vid_stats,
                })
                all_raw.extend(raw)

            print(f"  Total raw comments collected: {len(all_raw)}")

            if not all_raw:
                return jsonify({"error": "No comments found across any video in this playlist. Comments may be disabled."}), 400

            all_analysed = analyse_comments(all_raw)
            stats        = build_stats(all_analysed)

            return jsonify({
                "type":       "playlist",
                "model_used": "Custom Trained Model (Logistic Regression + TF-IDF)",
                "playlist": {
                    "title":       info["title"],
                    "channel":     info["channel"],
                    "thumbnail":   info["thumbnail"],
                    "video_count": len(video_ids),
                    "url":         f"https://youtube.com/playlist?list={entity_id}",
                },
                "stats":     stats,
                "per_video": per_video,
            })

        # ── SINGLE VIDEO ──
        else:
            info = get_video_info(youtube, entity_id)
            if not info:
                return jsonify({"error": "Video not found or is private."}), 404

            raw = collect_comments(youtube, entity_id, max_pages=3)
            print(f"  Raw comments collected: {len(raw)}")

            if not raw:
                return jsonify({"error": "No comments found. Comments may be disabled on this video."}), 400

            analysed = analyse_comments(raw)
            stats    = build_stats(analysed)

            return jsonify({
                "type":       "video",
                "model_used": "Custom Trained Model (Logistic Regression + TF-IDF)",
                "video": {
                    "title":     info["title"],
                    "channel":   info["channel"],
                    "thumbnail": info["thumbnail"],
                    "views":     info["views"],
                    "likes":     info["likes"],
                    "url":       f"https://youtube.com/watch?v={entity_id}",
                },
                "stats": stats,
            })

    except HttpError as e:
        msg = f"YouTube API error: {e.reason if hasattr(e,'reason') else str(e)}"
        print(f"  [ANALYSE ERROR] {msg}\n{traceback.format_exc()}")
        return jsonify({"error": msg}), 500
    except Exception as e:
        print(f"  [ANALYSE ERROR] {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)