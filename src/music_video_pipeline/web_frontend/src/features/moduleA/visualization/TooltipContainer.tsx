import React, { createContext, useCallback, useContext, useState } from "react";

type TooltipContent = string | null;

interface TooltipState {
  content: TooltipContent;
  x: number;
  y: number;
  visible: boolean;
}

interface TooltipContextValue {
  showTooltip: (content: string, x: number, y: number) => void;
  hideTooltip: () => void;
}

const TooltipContext = createContext<TooltipContextValue>({
  showTooltip: () => {},
  hideTooltip: () => {},
});

export function useTooltip() {
  return useContext(TooltipContext);
}

export function TooltipProvider({ children }: { children: React.ReactNode }) {
  const [state, setState] = useState<TooltipState>({
    content: null,
    x: 0,
    y: 0,
    visible: false,
  });

  const showTooltip = useCallback((content: string, x: number, y: number) => {
    setState({ content, x, y, visible: true });
  }, []);

  const hideTooltip = useCallback(() => {
    setState((prev) => ({ ...prev, visible: false }));
  }, []);

  return (
    <TooltipContext.Provider value={{ showTooltip, hideTooltip }}>
      {children}
      {state.visible && state.content && (
        <div
          className="vis-tooltip"
          style={{ left: state.x + 12, top: state.y + 12 }}
        >
          {state.content.split("\n").map((line, i) => {
            const colonIdx = line.indexOf(":");
            if (colonIdx > 0) {
              return (
                <div key={i}>
                  <span className="vis-tooltip__key">{line.slice(0, colonIdx + 1)}</span>
                  <span className="vis-tooltip__val">{line.slice(colonIdx + 1)}</span>
                </div>
              );
            }
            return <div key={i}>{line}</div>;
          })}
        </div>
      )}
    </TooltipContext.Provider>
  );
}
