
const jwt = require("jsonwebtoken");
const User = require("../models/User");

async function resolveUser(req, res, next, required) {
  try {
    const authorization = req.headers.authorization;

    if (!authorization) {
      if (!required) return next();

      return res.status(401).json({
        success: false,
        message: "Authentication required",
      });
    }

    const [scheme, token, extra] =
      authorization.split(" ");

    if (
      scheme !== "Bearer" ||
      !token ||
      extra
    ) {
      return res.status(401).json({
        success: false,
        message: "Invalid authorization header",
      });
    }

    const decoded = jwt.verify(
      token,
      process.env.JWT_SECRET
    );

    const user = await User.findById(decoded.id);

    if (!user) {
      return res.status(401).json({
        success: false,
        message: "User not found",
      });
    }

    req.user = user;
    next();
  } catch (error) {
    return res.status(401).json({
      success: false,
      message: "Invalid or expired token",
    });
  }
}

exports.requireAuth = (req, res, next) =>
  resolveUser(req, res, next, true);

exports.optionalAuth = (req, res, next) =>
  resolveUser(req, res, next, false);
