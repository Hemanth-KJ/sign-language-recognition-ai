
const AI_SERVICE_URL = (
  process.env.AI_SERVICE_URL || "http://127.0.0.1:8000"
).replace(/\/$/, "");

async function callAIService(path, options = {}) {
  const controller = new AbortController();

  const timeout = setTimeout(() => {
    controller.abort();
  }, 15000);

  try {
    const response = await fetch(
      `${AI_SERVICE_URL}${path}`,
      {
        ...options,
        signal: controller.signal,
        headers: {
          "Content-Type": "application/json",
          ...(options.headers || {}),
        },
      }
    );

    const data = await response.json();

    if (!response.ok) {
      const error = new Error(
        typeof data.detail === "string"
          ? data.detail
          : "AI service request failed"
      );

      error.status = response.status;
      throw error;
    }

    return data;
  } catch (error) {
    if (error.name === "AbortError") {
      const timeoutError = new Error(
        "AI service request timed out"
      );
      timeoutError.status = 504;
      throw timeoutError;
    }

    if (!error.status) {
      error.status = 503;
    }

    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

async function getMovementHealth() {
  return callAIService("/api/movement/health");
}

async function predictMovement(payload) {
  return callAIService("/api/movement/predict", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

async function resetMovement(payload) {
  return callAIService("/api/movement/reset", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}
async function predictMovementImage(payload) {
  return callAIService("/api/movement/predict-image", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

module.exports = {
  getMovementHealth,
  predictMovement,
  predictMovementImage,
  resetMovement,
};
