import type { PropsWithChildren } from "react";
import { useMemo } from "react";

import { App as AntdApp, ConfigProvider, theme } from "antd";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

export function AppProviders({ children }: PropsWithChildren) {
  const queryClient = useMemo(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            refetchOnWindowFocus: false,
            retry: 1,
            staleTime: 15_000,
          },
          mutations: {
            retry: 0,
          },
        },
      }),
    [],
  );

  return (
    <ConfigProvider
      theme={{
        algorithm: theme.defaultAlgorithm,
        token: {
          borderRadius: 8,
          colorPrimary: "#1768ac",
          colorInfo: "#1768ac",
          colorSuccess: "#287d3c",
          colorWarning: "#c07a00",
          colorError: "#c0392b",
          fontFamily:
            "\"PingFang SC\", \"Microsoft YaHei\", \"Noto Sans CJK SC\", sans-serif",
        },
      }}
    >
      <QueryClientProvider client={queryClient}>
        <AntdApp>{children}</AntdApp>
      </QueryClientProvider>
    </ConfigProvider>
  );
}
