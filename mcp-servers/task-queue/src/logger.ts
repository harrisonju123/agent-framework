import winston from "winston";
import { existsSync, mkdirSync } from "fs";
import { dirname } from "path";

export function createLogger() {
  const logPath = process.env.LOG_PATH || "./logs/mcp-task-queue.log";
  const logDir = dirname(logPath);

  if (!existsSync(logDir)) {
    mkdirSync(logDir, { recursive: true });
  }

  return winston.createLogger({
    level: process.env.LOG_LEVEL || "info",
    format: winston.format.combine(
      winston.format.timestamp(),
      winston.format.errors({ stack: true }),
      winston.format.json()
    ),
    transports: [
      new winston.transports.File({
        filename: logPath,
        maxsize: 10485760,
        maxFiles: 5,
      }),
    ],
  });
}
