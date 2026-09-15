CREATE TABLE IF NOT EXISTS "LiteLLM_AutoRouterBaselineState" (
    "scope" TEXT NOT NULL,
    "state" TEXT NOT NULL,
    "revision" BIGINT NOT NULL,
    CONSTRAINT "LiteLLM_AutoRouterBaselineState_pkey" PRIMARY KEY ("scope")
);
