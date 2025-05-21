const stream = require("stream");
const { promisify } = require("util");

process.env.GOOGLE_APPLICATION_CREDENTIALS = "google.json";


async function timeFunction(text) {
  const currentDate = new Date();
  console.log(text + currentDate.toISOString());
}


async function googleTextToWav(voiceName, text, speech_client) {
  timeFunction("Text to wav Starting: ");
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
    timeFunction("Text to wav ending: ");
    return audioContent;
  } catch (error) {
    console.error("Error in Google TTS:", error);
  }
}

module.exports = { googleTextToWav };
