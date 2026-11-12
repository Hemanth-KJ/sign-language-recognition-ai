
const express = require("express");

const {
  movementHealth,
  movementPredict,
  movementPredictImage,
  movementReset,
} = require("../controllers/movementController");

const router = express.Router();

router.get("/health", movementHealth);

router.post("/predict", movementPredict);

router.post("/predict-image", movementPredictImage);

router.post("/reset", movementReset);

module.exports = router;
