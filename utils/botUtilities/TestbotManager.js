const textToSpeech = require("@google-cloud/text-to-speech");
const { Pinecone } = require("@pinecone-database/pinecone");
const { OpenAIEmbeddings } = require("@langchain/openai");
const {
  createChain,
  processChat,
  parseConversationHistory,
} = require("./agent_chatbot");
const tts = require("./tts");
const dotenv = require("dotenv");
const speech = require("@google-cloud/speech");
const { Milvus } = require("@langchain/community/vectorstores/milvus");
const googleConfig = require("../../google.json");


// Load environment variables from .env file
dotenv.config();
const embeddings = new OpenAIEmbeddings({
  apiKey: process.env.OPENAI_API_KEY,
  model: "text-embedding-3-small",
});

// Milvus connection constants
const ZILLIZ_CLOUD_URI = "https://in03-6078dd857cb819c.serverless.gcp-us-west1.cloud.zilliz.com";
const ZILLIZ_CLOUD_USERNAME = "db_6078dd857cb819c";
const ZILLIZ_CLOUD_PASSWORD = "Sr7|FlM9lEK4BF4d";

const speechClient = new speech.SpeechClient({
  projectId: googleConfig.project_id,
  credentials: {
    private_key: googleConfig.private_key,
    client_email: googleConfig.client_email,
  },
});
// Language and voice mappings
const languageCodeMap = {
  English: "en-US",
  Spanish: "es-ES",
  French: "fr-FR",
  Russian: "ru-RU",
};

const voiceMap = {
  "English-Female": "en-US-Neural2-C",
  "English-Male": "en-GB-News-K",
  "Russian-Female": "ru-RU-Standard-A",
  "Russian-Male": "ru-RU-Standard-B",
  "French-Female": "fr-FR-Standard-C",
  "French-Male": "fr-FR-Standard-B",
  "Spanish-Female": "es-ES-Standard-A",
  "Spanish-Male": "es-ES-Standard-B",
};

// System prompts
const outboundSystemPrompt = `
 You are an intelligent call agent for outbound calls. The company will provide you with 'eligibility criteria,' 'restrictions', and 'end requirements'. \
Your task is to navigate the conversation based on what the user says, collecting any needed details. \
Make sure to use different sentence structures each time, and review the 'chat_history' to avoid repeating the same sentences at the end. \
Stick strictly to the given 'context', and do not introduce information beyond what is stated.
Context: {context}
Chat History: {chat_history}
**RESTRICTIONS:**
1. Strictly limit response to the exact information found in the 'Context'; do not add, assume, infer, or present any suggestions, advice, service details, timings, or other details not provided in 'Context'. If information is not available in 'Context', politely *INFORM* that you do not have information about it.
2. If there is any alternative information related to user query is in 'Context', provide that information to the user without adding additional information based on assumptions.
3. Once the conversation has started, do not repeat an introduction or restart the conversation from the 'chat history.'
4. Always review 'chat history' to determine what information has already been asked and provided. Do not offer further assistance or asking for "end requirements" repeatedly at the end of response if already offer/ask in 'chat history'.
5. Do not ask for information out of order or in a way that disregards the user's current question or stage in the process.
6. Do not request or collect information that is not explicitly specified to collect in the 'eligibility criteria', 'restrictions', 'end requirements', or 'context' like personal information or any other information etc under any circumstances.
7. If you ask the user that if user has any further queries, wait for their response before making any closing remarks like 'Have a great day' or 'Bye' or somewhat similar to it.
8. When all 'end requirements' have been gathered, do not restart from the beginning; instead, proceed to address any remaining user queries or bring the conversation to a close.
9. Ensure the 'eligibility criteria' are met as early as possible in the conversation. If not met, politely apologize and offer further assistance.
10. Maintain a natural, conversational tone, as a human agent would, and only greet the user once at the start of the conversation.
11. Handle restrictions by asking questions from 'end requirements' one by one, making sure to avoid asking previously answered questions.
12. If the user expresses disinterest, ask if they have any further queries before concluding the conversation.
13. Your response must be exclusively in {language} Language.
14. Your response must use the SSML <say-as> tag exclusively for formatting phone numbers(interpret as telephone), dates(interpret as date), currency values(interpret as cardinal), and characters(interpret as characters) etc.
`;

const inboundSystemPrompt = `
You are an intelligent call agent for inbound calls. You will be given the company's 'eligibility criteria,' and 'restrictions'. \
   Your primary task is to provide response to user query while strictly adhering to the 'Context', and ensuring no information is added beyond what is explicitly mentioned in 'Context'.\
   Each section is delimeted by Sequence of '*'.
   Greeting Message:
   {Greeting_message}
   ********************************************************************************************************
   Guidelines:
   {Restrictions}
   ********************************************************************************************************
   Eligibility Criteria:
   {Eligibility_criteria}
   ********************************************************************************************************
   Context from which Chatbot should answer: {context}
   ********************************************************************************************************
   Chat History: {chat_history}
   ********************************************************************************************************
   **RESTRICTIONS:**
   1. Maintain a natural, conversational tone, as a human agent would, and only greet the user once at the start of the conversation.
   2. Strictly limit response to the exact information found in the 'Context'; do not add, assume, infer, or present any suggestions, advice, service details, timings, or other details not provided in 'Context'.\
   If information is not available in 'Context' or 'chat_history' , politely *INFORM* that you do not have information about it.
   3. If the user expresses interest in the services, wants to make a reservation, booking an appointment or has other related queries, politely ask for their name, phone number, and any other required details based on the domain-specific 'end requirements' before proceeding further.
   4. Do not restart the conversation from the 'chat history' and always review 'chat history' to determine what information has already been asked and provided.
   5. When asking if the user needs further assistance, ensure that the statement varies each time to avoid repetition. Use different phrasings such as *"Would you like help with anything else?"*, *"Is there anything else I can assist you with today?"*, *"Do you have any other questions I can help with?"*, *"Let me know if there's anything else you need help with."*, or *"Would you like any additional information or assistance?"*. These variations should be used naturally throughout the conversation to maintain a more engaging and dynamic interaction.
   6. Do not ask for information out of order or in a way that disregards the user's current question or stage in the process.
   7. Do not request or collect information that is not explicitly specified to collect in the 'eligibility criteria', 'restrictions', or 'context' like personal information or any other information etc. under any circumstances.
   8. Do not acknowledge or refer to general system information like 'eligibility criteria,' 'restrictions,' or 'greeting messages' unless directly relevant to the user's query. Focus on addressing the user's question without mentioning the system setup.
   9. If the user does not meet the 'eligibility criteria', politely inform them and then ask if the user needs any further assistance rather than ending the conversation immediately.
   10. Restrict chatbot to same language which is  {language}  Language.
   11. You must not independently suggest actions i.e. scheduling appointments, placing orders, or any other activities unless explicitly stated in the 'Context'.
   12. Ask for reason of calling once if its not explicitly stated or mentioned  inside the 'chat_history'.
   13. Avoid premature conclusions. After responding to a query, ask if the user needs any further assistance. Wait for the user to indicate they are satisfied before making closing remarks like 'Have a great day' or 'Goodbye.'
   14. Understand the current query asked by the user and give responses according to that query.
`;

// Connection dictionary to store session information
let dictConnection = {};

// Helper function to log time
async function timeFunction(text) {
  const currentDate = new Date();
  console.log(text + currentDate.toISOString());
}

function initializeTestbotNamespace(namespace) {
  namespace.on("connect", (socket) => {
    console.log("New client connected to testbot");
    dictConnection[socket.id] = {
      connection: false,
      firstMessageCheck: true,
      delimeterCheck: false,
      conversationHistory: [],
      lastMessage: "",
    };

    socket.on("intial_data", async (data) => {
      if (data.additionalData) {
        const {
          Company_introduction,
          Greeting_message,
          Eligibility_criteria,
          End_requirements,
          Restrictions,
          voice_type,
          language,
        } = data.additionalData;

        let User_prompt;
        let SystemPrompt;
        if (data?.additionalData?.agent_type === "Make Calls") {
          User_prompt = `
          Company Introduction:
          ${Company_introduction}

          Greeting Message:
          ${Greeting_message}

          Eligibility Criteria:
          ${Eligibility_criteria}

          End Requirements:
          ${End_requirements}

          Restrictions:
          ${Restrictions}
          `;

          SystemPrompt = outboundSystemPrompt;
        } else if (data?.additionalData?.agent_type === "Answer Calls") {
          SystemPrompt = inboundSystemPrompt;
        }

        let vectorStore;
        if (data.additionalData.pinecone_index) {
          console.log("Pinecone Index");
          const pinecone = new Pinecone({
            apiKey: process.env.PINECONE_API_KEY,
          });

          // Get Pinecone index and namespace
          const pineconeIndex = pinecone.Index(data.additionalData.pinecone_index);
          const namespace = data.additionalData.pinecone_namespace;

          // Create custom retriever that properly formats results
          const customRetriever = {
            // Store the vector store information
            _vectorStore: {
              pineconeIndex,
              namespace,
              embeddings
            },
            // Method to create a retriever with options
            asRetriever: function(options) {
              this._searchOptions = options || {};
              return this;
            },
            // Method to get relevant documents
            getRelevantDocuments: async function(query) {
              try {
                console.log(`Searching Pinecone for: ${query}`);
                
                // Generate query embedding
                const queryEmbedding = await embeddings.embedQuery(query);
                
                // Set up query parameters
                const queryParams = {
                  vector: queryEmbedding,
                  topK: this._searchOptions?.search_kwargs?.k || 3,
                  includeMetadata: true,
                };
                
                // Query Pinecone directly
                let results;
                if (this._vectorStore.namespace) {
                  // Query with namespace if specified
                  results = await this._vectorStore.pineconeIndex
                    .namespace(this._vectorStore.namespace)
                    .query(queryParams);
                } else {
                  // Query without namespace
                  results = await this._vectorStore.pineconeIndex.query(queryParams);
                }
                
                console.log(`Found ${results.matches.length} matches in Pinecone`);
                
                // Format results to match document structure
                return results.matches.map(match => {
                  // Extract content from metadata or use fallback
                  const content = match.metadata?.text || 
                                 match.metadata?.content || 
                                 match.metadata?.pageContent || 
                                 JSON.stringify(match.metadata) || 
                                 'No content available';
                  
                  return {
                    pageContent: content,
                    metadata: {...match.metadata, score: match.score}
                  };
                });
              } catch (error) {
                console.error('Error in custom Pinecone retriever:', error);
                return []; // Return empty array on error
              }
            }
          };

          dictConnection[socket.id].vectorStore = customRetriever;
          dictConnection[socket.id].voice_type = voice_type;
          dictConnection[socket.id].index_type = "pinecone";
          dictConnection[socket.id].language = language;
          dictConnection[socket.id].languageCode =
            languageCodeMap[language] || "en-US";
          let formData = {
            Greeting_message,
            Eligibility_criteria,
            Restrictions,
            type: data?.additionalData?.agent_type,
          };
          dictConnection[socket.id].formData = formData;
          const chain = await createChain(SystemPrompt, User_prompt, formData);
          dictConnection[socket.id].chain = chain;
        } else if (data.additionalData.milvus_index) {
          console.log("Milvus Index");
          vectorStore = await Milvus.fromExistingCollection(embeddings, {
            collectionName: `_${data.additionalData.milvus_index}`,
            url: ZILLIZ_CLOUD_URI,
            username: ZILLIZ_CLOUD_USERNAME,
            password: ZILLIZ_CLOUD_PASSWORD,
          });

          dictConnection[socket.id].vectorStore = vectorStore;
          dictConnection[socket.id].voice_type = voice_type;
          dictConnection[socket.id].index_type = "milvus";
          dictConnection[socket.id].language = language;
          dictConnection[socket.id].languageCode =
            languageCodeMap[language] || "en-US";
          const chain = await createChain(SystemPrompt, User_prompt);
          dictConnection[socket.id].chain = chain;
        }
      }

      const client = new textToSpeech.TextToSpeechClient();
      dictConnection[socket.id].speech_model = client;
    });

    async function handleCommunication(socket, message) {
      const convHistory = parseConversationHistory(
        dictConnection[socket.id]?.conversationHistory
      );

      try {
        for await (const audioChunk of getResponse(
          socket.id,
          dictConnection[socket.id]?.chain,
          message,
          convHistory,
          dictConnection[socket.id]?.vectorStore,
          dictConnection[socket.id]?.index_type,
          dictConnection[socket.id]?.language,
          dictConnection[socket.id]?.formData
        )) {
          timeFunction("Last chunk sended Time: ");
          socket.emit("audio_chunk", audioChunk);
        }

        dictConnection[socket.id].conversationHistory.push({
          query: message,
          response: dictConnection[socket.id].lastMessage.trim(),
        });

        dictConnection[socket.id].lastMessage = "";
        dictConnection[socket.id].delimeterCheck = false;
      } catch (error) {
        console.error("Exception:", error);
      }
    }

    let recognizeStream = null;
    let finalText = "";
    let silenceTimeout = null;
    let voiceReceive = true;
    let botProcessing = false;
    socket.on("audio_chunk", async (data) => {
      let messageType = data?.type;

      if (messageType === "text") {
        console.log(Date.now());
        timeFunction("Message Receive Time: ");
        await handleCommunication(socket, data?.data);
      } else if (messageType === "audio") {
        if (voiceReceive) {
          timeFunction("First voice chunk received: ");
          voiceReceive = false;
        }
        try {
          if (!recognizeStream) {
            recognizeStream = speechClient
              .streamingRecognize({
                config: {
                  encoding: "LINEAR16",
                  sampleRateHertz: 16000,
                  languageCode: dictConnection[socket.id].languageCode,
                  enableWordTimeOffsets: true,
                  enableAutomaticPunctuation: true,
                  enableWordConfidence: true,
                  enableSpeakerDiarization: true,
                  model: "command_and_search",
                  useEnhanced: true,
                },
                interimResults: true,
              })
              .on("error", (err) => {
                stopRecognitionStream();
              })
              .on("data", (data2) => {
                const result = data2.results[0];
                const isFinal = result.isFinal;

                const transcription = data2.results
                  .map((result) => result.alternatives[0].transcript)
                  .join("\n");

                if (silenceTimeout) {
                  clearTimeout(silenceTimeout);
                }

                if (isFinal) {
                  finalText += transcription + " ";
                  stopRecognitionStream();
                }

                if (finalText.trim() === "") return;
                timeFunction("Voice Conversion Ended: ");
                silenceTimeout = setTimeout(async () => {
                  socket.emit("audio_converted", "Audio Converted");
                  timeFunction("Voice Delay Ended: ");
                  console.log(finalText, "User Text");
                  if (!botProcessing) {
                    botProcessing = true;
                    await handleCommunication(socket, finalText);
                    botProcessing = false;
                  }
                  finalText = "";
                }, 500);
              });
          }
          const audioBuffer = Buffer.from(data?.data);
          recognizeStream.write(audioBuffer);
        } catch (err) {
          console.error("Error processing audio data:", err);
        }
      }
    });

    function stopRecognitionStream() {
      if (recognizeStream) {
        recognizeStream.end();
      }
      recognizeStream = null;
    }

    socket.on("disconnect", () => {
      console.log("Client disconnected from testbot");
      stopRecognitionStream();
      delete dictConnection[socket.id];
    });
  });
}

async function* getResponse(
  sid,
  chain,
  query,
  chatHistory,
  vectorStore,
  indexType,
  language,
  formData
) {
  const responseGenerator = processChat(
    chain,
    query,
    chatHistory,
    vectorStore,
    indexType,
    language,
    formData
  );
  let collectedStr = "";
  const userLanguage = dictConnection[sid].language;
  const userVoiceType = dictConnection[sid].voice_type;
  const voiceKey = `${userLanguage}-${userVoiceType}`;
  const googleVoice = voiceMap[voiceKey];
  speech_model = dictConnection[sid].speech_model;

  for await (const value of responseGenerator) {
    if (value) {
      collectedStr += value;
    }

    if (
      value == "." ||
      value == "!" ||
      value == "?" ||
      value == "," ||
      value.includes("!") ||
      value.includes("?")
    ) {
      timeFunction("Chat Bot Ending Time: ");
      console.log("Bot Msg: ", collectedStr);
      yield await tts.googleTextToWav(
        googleVoice,
        collectedStr.trim().replace(/\*\*/g, ""),
        speech_model
      );
      dictConnection[sid].lastMessage += collectedStr + " ";
      collectedStr = "";
    }
  }
}

module.exports = { initializeTestbotNamespace };
