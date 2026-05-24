import { Outlet, useLocation } from "react-router-dom";

export function TaskShell() {
  const location = useLocation();
  const isReviewPage = location.pathname.endsWith("/review");

  return (
    <div className={`task-shell ${isReviewPage ? "task-shell--review" : ""}`}>
      <div className="task-shell__outlet">
        <Outlet />
      </div>
    </div>
  );
}
