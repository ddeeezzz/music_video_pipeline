import { Empty } from "antd";

type EmptyStateProps = {
  description: string;
};

export function EmptyState({ description }: EmptyStateProps) {
  return <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={description} />;
}
