
const express = require("express");
const controller = require("../controllers/signController");

const {
  requireAuth,
} = require("../middleware/authMiddleware");

const router = express.Router();

router.get("/", controller.list);

router.post("/", requireAuth, controller.create);

module.exports = router;
