
const express = require("express");
const cors = require("cors");
const mongoose = require("mongoose");
require("dotenv").config();

const connectDB = require("./config/db");

const authRoute = require("./routes/auth");
const signRoute = require("./routes/sign");
const predictRoute = require("./routes/predict");
const movementRoutes = require("./routes/movementRoutes");

const app = express();

const PORT = Number(process.env.PORT || 5000);

const allowedOrigins = (
  process.env.CLIENT_URL ||
  "http://localhost:5173,http://127.0.0.1:5173"
)
  .split(",")
  .map((origin) => origin.trim());

// ==========================
// MIDDLEWARE
// ==========================

app.use(
  cors({
    origin(origin, callback) {
      if (!origin || allowedOrigins.includes(origin)) {
        return callback(null, true);
      }

      return callback(new Error("Origin not allowed by CORS"));
    },
    credentials: true,
  })
);

app.use(
  express.json({
    limit: "10mb",
  })
);

app.use(
  express.urlencoded({
    extended: true,
    limit: "1mb",
  })
);

app.use("/api/movement", movementRoutes);

// ==========================
// HEALTH CHECK
// ==========================

app.get("/", (req, res) => {
  return res.json({
    success: true,
    message: "SignAI Node.js backend running",
    database:
      mongoose.connection.readyState === 1
        ? "connected"
        : "disconnected",
  });
});

// ==========================
// ROUTES
// ==========================

app.use("/auth", authRoute);
app.use("/sign", signRoute);
app.use("/predict", predictRoute);

// ==========================
// 404 HANDLER
// ==========================

app.use((req, res) => {
  return res.status(404).json({
    success: false,
    message: "Route not found",
  });
});

// ==========================
// ERROR HANDLER
// ==========================

app.use((error, req, res, next) => {
  console.error("Server error:", error.message);

  if (error.type === "entity.too.large") {
    return res.status(413).json({
      success: false,
      message: "Request body too large",
    });
  }

  if (error instanceof SyntaxError && "body" in error) {
    return res.status(400).json({
      success: false,
      message: "Invalid JSON",
    });
  }

  if (error.message === "Origin not allowed by CORS") {
    return res.status(403).json({
      success: false,
      message: "Origin not allowed",
    });
  }

  return res.status(500).json({
    success: false,
    message: "Internal server error",
  });
});


// ==========================
// START SERVER
// ==========================

async function startServer() {
  try {
    if (!process.env.JWT_SECRET) {
      throw new Error("JWT_SECRET missing in .env");
    }

    await connectDB();

    const server = app.listen(PORT, () => {
      console.log(`
=================================
SignAI Backend Running
http://localhost:${PORT}

MongoDB: Connected
AI Service: ${process.env.AI_SERVICE || "http://127.0.0.1:8000"}
=================================
`);
    });

    async function shutdown() {
      console.log("Shutting down SignAI backend...");

      server.close(async () => {
        await mongoose.connection.close();
        process.exit(0);
      });
    }

    process.once("SIGINT", shutdown);
    process.once("SIGTERM", shutdown);
  } catch (error) {
    console.error(
      "Server startup failed:",
      error.message
    );

    process.exit(1);
  }
}

startServer();
