type LogPayload = unknown;

function formatMessage(moduleName: string, message: string): string {
  return `[${moduleName}] ${message}`;
}

function writeLog(
  writer: (...args: unknown[]) => void,
  moduleName: string,
  message: string,
  payload?: LogPayload,
): void {
  const formattedMessage = formatMessage(moduleName, message);
  if (payload === undefined) {
    writer(formattedMessage);
    return;
  }
  writer(formattedMessage, payload);
}

export const appLogger = {
  info(moduleName: string, message: string, payload?: LogPayload): void {
    writeLog(console.info, moduleName, message, payload);
  },
  warn(moduleName: string, message: string, payload?: LogPayload): void {
    writeLog(console.warn, moduleName, message, payload);
  },
  error(moduleName: string, message: string, payload?: LogPayload): void {
    writeLog(console.error, moduleName, message, payload);
  },
};
