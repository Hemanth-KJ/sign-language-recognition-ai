
const express = require("express");
const controller = require("../controllers/predictController");

const {
  requireAuth,
  optionalAuth,
} = require("../middleware/authMiddleware");

const router = express.Router();

router.post("/", optionalAuth, controller.predictSign);

router.get(
  "/history",
  requireAuth,
  controller.getHistory
);

module.exports = router;
