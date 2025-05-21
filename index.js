const express = require("express");
const app = express();
const http = require("http");
const bodyParser = require("body-parser");
const dotenv = require("dotenv");
const cors = require("cors");
const {
  initializeTestbotNamespace,
} = require("./utils/botUtilities/TestbotManager");

const WebSocket = require("ws");

// Load environment variables from .env file
dotenv.config();

// Middleware and configurations
app.use(cors());
app.use(express.json());
app.use(bodyParser.json());
app.use(express.urlencoded({ extended: true }));

const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

const io = require("socket.io")(server, {
  cors: {
    origin: "*",
  },
});

initializeTestbotNamespace(io.of("/testbot"));


server.on("upgrade", (request, socket, head) => {
  const pathname = request.url;
  if (!pathname.startsWith("/callbot/")) {
    console.log("Socket destroyed");
    socket.destroy();
  }
});


// Jobs the scheduler
// Payment the scheduler



server.listen("8800", () => {
  console.log("Server is running perfectly on port 8800");
});
