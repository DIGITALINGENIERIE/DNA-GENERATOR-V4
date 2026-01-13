import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api, buildUrl, type AnalysisInput } from "@shared/routes";
import { useLocation } from "wouter";

export function useAnalysis(id?: number) {
  return useQuery({
    queryKey: [api.analyses.get.path, id],
    queryFn: async () => {
      if (!id) return null;
      const url = buildUrl(api.analyses.get.path, { id });
      const res = await fetch(url, { credentials: "include" });
      if (res.status === 404) return null;
      if (!res.ok) throw new Error("Failed to fetch analysis");
      return api.analyses.get.responses[200].parse(await res.json());
    },
    enabled: !!id,
    refetchInterval: (query) => {
      const status = query.state.data?.analysis.status;
      // Stop polling if completed or failed, otherwise poll every 2s
      return status === "completed" || status === "failed" ? false : 2000;
    },
  });
}

export function useCreateAnalysis() {
  const queryClient = useQueryClient();
  const [, setLocation] = useLocation();

  return useMutation({
    mutationFn: async (data: AnalysisInput) => {
      const validated = api.analyses.create.input.parse(data);
      const res = await fetch(api.analyses.create.path, {
        method: api.analyses.create.method,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(validated),
        credentials: "include",
      });

      if (!res.ok) {
        if (res.status === 400) {
          const error = api.analyses.create.responses[400].parse(await res.json());
          throw new Error(error.message);
        }
        throw new Error("Failed to initiate analysis sequence");
      }
      return api.analyses.create.responses[201].parse(await res.json());
    },
    onSuccess: (data) => {
      // Invalidate list queries if we had them
      // Navigate to the analysis page
      setLocation(`/analysis/${data.id}`);
    },
  });
}
