import { Navigate, createBrowserRouter } from "react-router-dom";

import { routes } from "@/app/routes";
import { AppShell } from "@/components/layout/AppShell";
import { TaskShell } from "@/components/layout/TaskShell";
import { TaskCreatePage } from "@/pages/tasks/TaskCreatePage";
import { TaskDetailPage } from "@/pages/tasks/TaskDetailPage";
import { TaskListPage } from "@/pages/tasks/TaskListPage";
import { TaskModuleAPage } from "@/pages/tasks/TaskModuleAPage";
import { TaskModuleBPage } from "@/pages/tasks/TaskModuleBPage";
import { TaskMonitorPage } from "@/pages/tasks/TaskMonitorPage";
import { TaskReviewPage } from "@/pages/tasks/TaskReviewPage";

function NotFoundRedirect() {
  return <Navigate replace to={routes.taskList} />;
}

export const router = createBrowserRouter([
  {
    path: "/",
    element: <Navigate replace to={routes.taskList} />,
  },
  {
    path: "/tasks",
    element: <AppShell />,
    children: [
      {
        index: true,
        element: <TaskListPage />,
      },
      {
        path: "create",
        element: <TaskCreatePage />,
      },
      {
        path: ":taskId",
        element: <TaskShell />,
        children: [
          {
            index: true,
            element: <TaskDetailPage />,
          },
          {
            path: "monitor",
            element: <TaskMonitorPage />,
          },
          {
            path: "review",
            element: <TaskReviewPage />,
          },
          {
            path: "module-a",
            element: <TaskModuleAPage />,
          },
          {
            path: "module-b",
            element: <TaskModuleBPage />,
          },
        ],
      },
    ],
  },
  {
    path: "*",
    element: <NotFoundRedirect />,
  },
]);
