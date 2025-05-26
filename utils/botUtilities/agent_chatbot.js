const {
  ChatOpenAI,
  OpenAIEmbeddings,
  AzureChatOpenAI,
} = require("@langchain/openai");
const {
  ChatPromptTemplate,
  MessagesPlaceholder,
} = require("@langchain/core/prompts");
const {
  createStuffDocumentsChain,
} = require("langchain/chains/combine_documents");
const {
  HumanMessage,
  AIMessage,
} = require("@langchain/core/messages");
const dotenv = require("dotenv");

// Load environment variables from .env file
dotenv.config();

async function timeFunction(text) {
  const currentDate = new Date();
  console.log(text + currentDate.toISOString());
}

// Cache for vector search results to avoid redundant searches
const vectorSearchCache = new Map();
const CACHE_TTL = 10 * 60 * 1000; // 10 minutes cache TTL
const TOP_K = 2; // Reduced from default (usually 3-10) for speed

// Helper to get cache key
function getCacheKey(question, indexType) {
  return `${indexType}:${question.trim().toLowerCase()}`;
}

async function* processChat(
  chain,
  question,
  chatHistory,
  vectorStore,
  indexType,
  language,
  formData
) {
  // Start timing for the entire process
  const totalStartTime = Date.now();
  console.log(`PROCESS START: ${new Date().toISOString()}`);
  
  // Start timing for vector retrieval
  const retrievalStartTime = Date.now();
  let docs;
  
  // Start vector search timing
  const vectorSearchStartTime = Date.now();
  
  // Check if we have a cached result for this question
  const cacheKey = getCacheKey(question, indexType);
  const cachedResult = vectorSearchCache.get(cacheKey);
  
  if (cachedResult && (Date.now() - cachedResult.timestamp < CACHE_TTL)) {
    docs = cachedResult.docs;
    console.log('Using cached vector search result');
    const vectorSearchTime = Date.now() - vectorSearchStartTime;
    console.log(`VECTOR CACHE HIT TIME: ${(vectorSearchTime/1000).toFixed(2)} sec`);
  } else {
    // Optimize vector search based on index type
    if (indexType === "pinecone") {
      console.log("pinecone retriever working");
      const searchStartTime = Date.now();
      
      // Set up optimized query parameters
      const searchOptions = { 
        search_kwargs: { 
          k: TOP_K,
          include_metadata: true,
          include_values: false // Don't need vectors back
        }
      };
      
      docs = await vectorStore.getRelevantDocuments(question, searchOptions);
      
      const searchTime = Date.now() - searchStartTime;
      console.log(`PINECONE SEARCH TIME: ${(searchTime/1000).toFixed(2)} sec`);
      console.log(`Retrieved ${docs.length} documents`);
      
    } else if (indexType === "milvus") {
      console.log("milvus retriever working");
      const searchStartTime = Date.now();
      
      // Use direct similarity search with optimized params
      docs = await vectorStore.similaritySearch(question, TOP_K, {
        includeValues: false, // Don't need vectors back
        consistencyLevel: "Eventually" // Faster reads
      });
      
      const searchTime = Date.now() - searchStartTime;
      console.log(`MILVUS SEARCH TIME: ${(searchTime/1000).toFixed(2)} sec`);
    }
    
    // Cache the results
    vectorSearchCache.set(cacheKey, {
      docs,
      timestamp: Date.now()
    });
    
    // Clean up old cache entries periodically
    if (vectorSearchCache.size > 100) {
      const now = Date.now();
      for (const [key, value] of vectorSearchCache.entries()) {
        if (now - value.timestamp > CACHE_TTL) {
          vectorSearchCache.delete(key);
        }
      }
    }
  }

  // End timing for vector retrieval
  const retrievalEndTime = Date.now();
  const retrievalTime = retrievalEndTime - retrievalStartTime;
  const retrievalSeconds = (retrievalTime / 1000).toFixed(2);
  console.log(`VECTOR RETRIEVAL TIME: ${retrievalSeconds} sec (${indexType})`);

  // Start timing for AI processing
  const aiStartTime = Date.now();
  
  // Always limit chat_history to last 5 messages (if array)
  let limitedHistory = Array.isArray(chatHistory)
    ? chatHistory.slice(-5)
    : chatHistory;

  let InpData = {
    chat_history: limitedHistory,
    input:
      language === "English"
        ? question
        : JSON.stringify(question.trim().replace(/\s+/g, " ").normalize("NFC")),
    context: docs, // do not truncate context
    language: language,
  };

  if (formData?.type === "Answer Calls") {
    InpData.Greeting_message = formData?.Greeting_message;
    InpData.Eligibility_criteria = formData?.Eligibility_criteria;
    InpData.Restrictions = formData?.Restrictions;
  }
  console.log("InpData", InpData);

  // Start LLM processing timing
  const llmStartTime = Date.now();
  
  // Stream with optimized parameters
  const response = await chain.stream({
    ...InpData,
    stream: true,
    timeout: 30000, // 30 second timeout
  });
  
  // Track streaming stats
  let chunkCount = 0;
  let firstChunkTime = null;
  
  for await (var chunk of response) {
    chunkCount++;
    if (chunkCount === 1) {
      firstChunkTime = Date.now();
      const timeToFirstChunk = firstChunkTime - llmStartTime;
      console.log(`TIME TO FIRST CHUNK: ${(timeToFirstChunk/1000).toFixed(2)} sec`);
    }
    yield chunk;
  }
  
  // End timing for AI processing
  const aiEndTime = Date.now();
  const aiTime = aiEndTime - aiStartTime;
  const aiSeconds = (aiTime / 1000).toFixed(2);
  console.log(`TOTAL LLM TIME: ${aiSeconds} sec (${chunkCount} chunks)`);
  
  if (firstChunkTime) {
    const streamingTime = aiEndTime - firstChunkTime;
    console.log(`STREAMING TIME: ${(streamingTime/1000).toFixed(2)} sec`);
  }
  
  // End timing for the entire process
  const totalEndTime = Date.now();
  const totalTime = totalEndTime - totalStartTime;
  const totalSeconds = (totalTime / 1000).toFixed(2);
  console.log(`TOTAL PROCESS TIME: ${totalSeconds} sec`);
}

function parseConversationHistory(history) {
  const tempHistory = [];
  history?.forEach((row) => {
    tempHistory.push(new HumanMessage({ content: row.query }));
    tempHistory.push(new AIMessage({ content: row.response }));
  });
  return tempHistory;
}

async function createChain(
  systemPrompt,
  userPrompt,
  formData,
  model = "azure"
) {
  try {
    let modelInstance;
    if (model === "gpt") {
      modelInstance = new ChatOpenAI({
        model: "gpt-3.5-turbo", // Faster than GPT-4
        temperature: 0.2,
        max_tokens: 256,
        presence_penalty: 0,
        frequency_penalty: 0,
      });
    } else if (model == "azure") {
      console.log("Using Azure Model: .......");
      modelInstance = new AzureChatOpenAI({
        temperature: 0.2,
        maxRetries: 1, // Reduced retries for faster failure
        max_tokens: 256,
        presence_penalty: 0,
        frequency_penalty: 0,
        azureOpenAIApiKey: process.env.AZURE_OPENAI_API_KEY_NAME,
      });
    }

    let prompt;

    if (formData?.type === "Make Calls") {
      prompt = ChatPromptTemplate.fromMessages([
        ["system", systemPrompt],
        ["system", userPrompt],
        new MessagesPlaceholder("chat_history"),
        ["user", "{input}"],
        ["user", "{language}"],
      ]);
    } else if (formData?.type === "Answer Calls") {
      prompt = ChatPromptTemplate.fromMessages([
        ["system", systemPrompt],
        new MessagesPlaceholder("chat_history"),
        new MessagesPlaceholder("Greeting_message"),
        new MessagesPlaceholder("Eligibility_criteria"),
        new MessagesPlaceholder("Restrictions"),
        ["user", "{input}"],
        ["user", "{language}"],
      ]);
    }

    // Chain creation using prompt and model
    const chain = await createStuffDocumentsChain({
      llm: modelInstance,
      prompt,
    });

    return chain;
  } catch (error) {
    console.error(error);
  }
}

module.exports = {
  createChain,
  processChat,
  parseConversationHistory,
};
