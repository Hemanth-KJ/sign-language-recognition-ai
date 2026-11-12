
const Sign = require("../models/SignModelTemp");

exports.create = async (req, res) => {
  try {
    const { name, description = "" } = req.body || {};

    if (
      typeof name !== "string" ||
      !name.trim() ||
      typeof description !== "string"
    ) {
      return res.status(400).json({
        success: false,
        message: "Valid sign name and description required",
      });
    }

    const sign = await Sign.create({
      name: name.trim(),
      description: description.trim(),
    });

    return res.status(201).json({
      success: true,
      sign,
    });
  } catch (error) {
    if (error.code === 11000) {
      return res.status(409).json({
        success: false,
        message: "Sign already exists",
      });
    }

    console.error("Create sign error:", error);

    return res.status(500).json({
      success: false,
      message: "Unable to create sign",
    });
  }
};

exports.list = async (req, res) => {
  try {
    const signs = await Sign.find()
      .sort({ name: 1 })
      .lean();

    return res.json({
      success: true,
      signs,
    });
  } catch (error) {
    console.error("List signs error:", error);

    return res.status(500).json({
      success: false,
      message: "Unable to load signs",
    });
  }
};
