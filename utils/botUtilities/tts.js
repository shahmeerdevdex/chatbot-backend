const stream = require("stream");
const { promisify } = require("util");
const crypto = require("crypto");
const fs = require("fs").promises;
const path = require("path");

process.env.GOOGLE_APPLICATION_CREDENTIALS = "google.json";

// Create cache directory if it doesn't exist
const CACHE_DIR = path.join(process.cwd(), "tts_cache");

async function ensureCacheDir() {
  try {
    await fs.mkdir(CACHE_DIR, { recursive: true });
  } catch (error) {
    console.error("Error creating cache directory:", error);
  }
}

// Initialize cache directory
ensureCacheDir();

async function timeFunction(text) {
  const currentDate = new Date();
}

// Generate a hash for the text and voice to use as cache key
function getCacheKey(voiceName, text) {
  return crypto
    .createHash("md5")
    .update(`${voiceName}:${text}`)
    .digest("hex");
}

async function googleTextToWav(voiceName, text, speech_client) {
  // Start timing for TTS
  const ttsStartTime = Date.now();
  
  // Check if we have this in cache
  const cacheKey = getCacheKey(voiceName, text);
  const cachePath = path.join(CACHE_DIR, `${cacheKey}.mp3`);
  
  try {
    // Try to read from cache first
    const stats = await fs.stat(cachePath);
    if (stats.isFile() && stats.size > 0) {
      const audioContent = await fs.readFile(cachePath);
      // End timing for TTS (cache hit)
      const ttsEndTime = Date.now();
      const ttsTotalTime = ttsEndTime - ttsStartTime;
      const ttsTotalSeconds = (ttsTotalTime / 1000).toFixed(2);
      console.log(`TTS TIME: ${ttsTotalSeconds} sec (cache hit)`);
      return audioContent;
    }
  } catch (error) {
    // File doesn't exist or other error, proceed with API call
  }

  // Not in cache, make the API call
  const request = {
    input: { ssml: `<speak>${text}</speak>` },
    voice: {
      languageCode: voiceName.split("-").slice(0, 2).join("-"),
      name: voiceName,
    },
    audioConfig: { audioEncoding: "MP3" },
  };

  try {
    const [response] = await speech_client.synthesizeSpeech(request);
    const audioContent = response.audioContent;
    
    // Save to cache for future use
    await fs.writeFile(cachePath, audioContent);
    
    // End timing for TTS (API call)
    const ttsEndTime = Date.now();
    const ttsTotalTime = ttsEndTime - ttsStartTime;
    const ttsTotalSeconds = (ttsTotalTime / 1000).toFixed(2);
    console.log(`TTS TIME: ${ttsTotalSeconds} sec (API call)`);
    
    return audioContent;
  } catch (error) {
    console.error("Error in Google TTS:", error);
    // End timing even in case of error
    const ttsEndTime = Date.now();
    const ttsTotalTime = ttsEndTime - ttsStartTime;
    const ttsTotalSeconds = (ttsTotalTime / 1000).toFixed(2);
    console.log(`TTS TIME: ${ttsTotalSeconds} sec (error)`);
  }
}

module.exports = { googleTextToWav };
