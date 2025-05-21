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

async function* processChat(
  chain,
  question,
  chatHistory,
  vectorStore,
  indexType,
  language,
  formData
) {
  timeFunction("Before Retriever: ");
  let docs;
  if (indexType === "milvus") {
    console.log("milvus retriever working");
    const retriever = vectorStore.asRetriever({
      searchType: "similarity",
      k: 3,
    });
    let newdocs = await retriever.getRelevantDocuments(question);
    docs = newdocs.map((doc) => ({
      ...doc,
      pageContent: doc.metadata?.text || doc.pageContent,
    }));
  } else {
    console.log("pinecone retriever working");
    const retriever = vectorStore.asRetriever({
      search_type: "mmr",
      search_kwargs: { k: 3 },
    });
    docs = await retriever.getRelevantDocuments(question);
  }

  timeFunction("After Retriever: ");
  timeFunction("Chat Bot Starting Time: ");

  let InpData = {
    chat_history: chatHistory,
    input:
      language === "English"
        ? question
        : JSON.stringify(question.trim().replace(/\s+/g, " ").normalize("NFC")),
    context: docs,
    language: language,
  };

  if (formData?.type === "Answer Calls") {
    InpData.Greeting_message = formData?.Greeting_message;
    InpData.Eligibility_criteria = formData?.Eligibility_criteria;
    InpData.Restrictions = formData?.Restrictions;
  }

  const response = await chain.stream(InpData);
    
  for await (var chunk of response) {
    yield chunk;
  }
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
        model: "gpt-4o",
        temperature: 0.1,
      });
    } else if (model == "azure") {
      console.log("Using Azure Model: .......");
      modelInstance = new AzureChatOpenAI({
        temperature: 0,
        maxRetries: 2,
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
