
const {
  getMovementHealth,
  predictMovement,
   predictMovementImage,
  resetMovement,
} = require("../services/movementService");

const UUID_REGEX =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

function handleError(res, error) {
  console.error("Movement API error:", error.message);

  const status =
    Number.isInteger(error.status) &&
    error.status >= 400 &&
    error.status < 600
      ? error.status
      : 500;

  return res.status(status).json({
    success: false,
    message: error.message,
  });
}

function validSessionId(sessionId) {
  return (
    typeof sessionId === "string" &&
    UUID_REGEX.test(sessionId)
  );
}

async function movementHealth(req, res) {
  try {
    const result = await getMovementHealth();
    return res.json(result);
  } catch (error) {
    return handleError(res, error);
  }
}

async function movementPredict(req, res) {
  try {
    const {
      sessionId,
      features,
      handDetected = true,
    } = req.body || {};

    if (!validSessionId(sessionId)) {
      return res.status(400).json({
        success: false,
        message: "A valid UUID sessionId is required",
      });
    }

    if (
      !Array.isArray(features) ||
      features.length !== 126 ||
      !features.every(
        (value) =>
          typeof value === "number" &&
          Number.isFinite(value)
      )
    ) {
      return res.status(400).json({
        success: false,
        message: "features must contain 126 finite numbers",
      });
    }

    if (typeof handDetected !== "boolean") {
      return res.status(400).json({
        success: false,
        message: "handDetected must be a boolean",
      });
    }

    const result = await predictMovement({
      sessionId,
      features,
      handDetected,
    });

    return res.json(result);
  } catch (error) {
    return handleError(res, error);
  }
}

async function movementPredictImage(req, res) {
  try {
    const { sessionId, image } = req.body || {};

    if (!validSessionId(sessionId)) {
      return res.status(400).json({
        success: false,
        message: "A valid sessionId is required",
      });
    }

    if (
      typeof image !== "string" ||
      image.length === 0 ||
      image.length > 7_000_000
    ) {
      return res.status(400).json({
        success: false,
        message: "A valid image is required (maximum 5 MB decoded)",
      });
    }

    const result = await predictMovementImage({
      sessionId,
      image,
    });

    return res.json(result);
  } catch (error) {
    return handleError(res, error);
  }
}

async function movementReset(req, res) {
  try {
    const { sessionId } = req.body || {};

    if (!validSessionId(sessionId)) {
      return res.status(400).json({
        success: false,
        message: "A valid UUID sessionId is required",
      });
    }

    const result = await resetMovement({
      sessionId,
    });

    return res.json(result);
  } catch (error) {
    return handleError(res, error);
  }
}

module.exports = {
  movementHealth,
  movementPredict,
  movementPredictImage,
  movementReset,
};
