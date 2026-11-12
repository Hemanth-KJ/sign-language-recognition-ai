
const axios = require("axios");
const Prediction = require("../models/Prediction");

const AI_SERVICE =
  process.env.AI_SERVICE ||
  "http://127.0.0.1:8000";

const AI_TIMEOUT = 20000;
const MAX_HISTORY = 50;

// Avoid storing identical predictions every frame.
// This is per user, or per anonymous session token.
const lastSaved = new Map();
const SAVE_COOLDOWN_MS = 10000;
const MAX_CACHE_ENTRIES = 1000;

function getSessionKey(req) {
  if (req.user) {
    return `user:${req.user._id}`;
  }

  const sessionId = req.headers["x-session-id"];

  if (
    typeof sessionId === "string" &&
    /^[a-zA-Z0-9_-]{8,100}$/.test(sessionId)
  ) {
    return `session:${sessionId}`;
  }

  return null;
}

function shouldSave(sessionKey, sign) {
  if (!sessionKey) return false;

  const key = `${sessionKey}:${sign}`;
  const now = Date.now();
  const previous = lastSaved.get(key);

  if (previous && now - previous < SAVE_COOLDOWN_MS) {
    return false;
  }

  if (lastSaved.size >= MAX_CACHE_ENTRIES) {
    lastSaved.clear();
  }

  lastSaved.set(key, now);
  return true;
}

exports.predictSign = async (req, res) => {
  try {
    const { image } = req.body || {};

    if (
      typeof image !== "string" ||
      !image.trim()
    ) {
      return res.status(400).json({
        success: false,
        message: "Image required",
      });
    }

    const response = await axios.post(
      `${AI_SERVICE.replace(/\/$/, "")}/predict`,
      { image },
      {
        timeout: AI_TIMEOUT,
        maxBodyLength: 10 * 1024 * 1024,
      }
    );

    const result = response.data;

    if (
      result?.success !== true ||
      !["detected", "no_hand", "low_confidence"]
        .includes(result.status)
    ) {
      return res.status(502).json({
        success: false,
        message: "Invalid AI service response",
      });
    }

    const prediction =
      result.status === "detected"
        ? String(result.prediction || "")
        : null;

    const confidence = Number(result.confidence);

    if (
      !Number.isFinite(confidence) ||
      confidence < 0 ||
      confidence > 100 ||
      (result.status === "detected" && !prediction)
    ) {
      return res.status(502).json({
        success: false,
        message: "Invalid prediction data",
      });
    }

    let saved = false;

    if (result.status === "detected") {
      const sessionKey = getSessionKey(req);

      if (shouldSave(sessionKey, prediction)) {
        try {
          await Prediction.create({
            user: req.user?._id || null,
            sign: prediction,
            confidence,
            classId: Number.isInteger(result.class_id)
              ? result.class_id
              : null,
            status: "detected",
          });

          saved = true;
        } catch (dbError) {
          console.error(
            "Prediction history save failed:",
            dbError.message
          );
        }
      }
    }

    return res.json({
      success: true,
      status: result.status,
      prediction,
      confidence,
      class_id: result.class_id ?? null,
      saved,
    });
  } catch (error) {
    console.error(
      "AI prediction error:",
      error.response?.data || error.message
    );

    if (error.response) {
      const status = error.response.status;

      return res.status(
        status >= 400 && status < 500 ? status : 502
      ).json({
        success: false,
        message:
          error.response.data?.detail ||
          "AI prediction service failed",
      });
    }

    return res.status(503).json({
      success: false,
      message:
        error.code === "ECONNABORTED"
          ? "AI prediction timed out"
          : "AI service unavailable",
    });
  }
};

exports.getHistory = async (req, res) => {
  try {
    const history = await Prediction.find({
      user: req.user._id,
    })
      .sort({ createdAt: -1 })
      .limit(MAX_HISTORY)
      .lean();

    return res.json({
      success: true,
      history,
    });
  } catch (error) {
    console.error("History error:", error);

    return res.status(500).json({
      success: false,
      message: "Unable to load prediction history",
    });
  }
};
