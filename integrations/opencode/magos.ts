import type { Plugin } from "@opencode-ai/plugin"
import type { Config } from "@opencode-ai/sdk"

const DEFAULT_BASE_URL = "http://localhost:6246"
const FETCH_TIMEOUT_MS = 3000

type MagosEntry = {
  provider: string
  raw_id: string
  litellm_id: string
  context_size: number | null
  max_output: number | null
  input_cost: number | null
  output_cost: number | null
  cache_read_cost: number | null
  cache_write_cost: number | null
  input_modalities: string[]
  output_modalities: string[]
  deprecated_at: string | null
  sources: string[]
}

type MagosRegistry = {
  refreshed_at: Record<string, string>
  entries: MagosEntry[]
}

type ProviderConfig = NonNullable<Config["provider"]>[string]
type ModelConfig = NonNullable<ProviderConfig["models"]>[string]
type Modality = "text" | "audio" | "image" | "video" | "pdf"

/**
 * Resolve the magos host root.
 *
 * Priority: `provider.magos.options.baseURL` from opencode.json >
 * `MAGOS_BASE_URL` env var > `DEFAULT_BASE_URL`. The opencode value
 * typically has a `/v1` suffix (the SDK uses it verbatim for chat
 * completions); we strip it so the same root works for
 * `/admin/registry`.
 */
function baseURL(cfg: ProviderConfig): string {
  const fromOptions = cfg.options?.baseURL
  const raw =
    (typeof fromOptions === "string" && fromOptions) || process.env.MAGOS_BASE_URL || DEFAULT_BASE_URL
  return raw.replace(/\/+$/, "").replace(/\/v1$/, "")
}

function namespacedId(entry: MagosEntry): string {
  return `${entry.provider}/${entry.raw_id}`
}

/**
 * Map magos modality strings onto opencode's supported set. Magos emits
 * `file` for arbitrary file uploads (PDFs, etc.); opencode only tracks
 * `pdf` specifically, so we fold one into the other.
 */
function toModalities(modalities: string[]): Modality[] {
  const out = new Set<Modality>()
  for (const m of modalities) {
    if (m === "text" || m === "audio" || m === "image" || m === "video") out.add(m)
    else if (m === "file") out.add("pdf")
  }
  return [...out]
}

function entryToModel(entry: MagosEntry): ModelConfig {
  const id = namespacedId(entry)
  const attachment = entry.input_modalities.some((m) => m !== "text")
  const cost: ModelConfig["cost"] = {
    input: entry.input_cost ?? 0,
    output: entry.output_cost ?? 0,
  }
  if (entry.cache_read_cost != null) cost.cache_read = entry.cache_read_cost
  if (entry.cache_write_cost != null) cost.cache_write = entry.cache_write_cost
  return {
    // The wire-level `model` field sent in the request body. Must match
    // the namespaced id so magos.yaml rules and the auto-router can match
    // against it.
    id,
    name: id,
    attachment,
    tool_call: true,
    temperature: true,
    cost,
    limit: {
      context: entry.context_size ?? 0,
      output: entry.max_output ?? entry.context_size ?? 0,
    },
    modalities: {
      input: toModalities(entry.input_modalities),
      output: toModalities(entry.output_modalities),
    },
  }
}

async function fetchRegistry(base: string): Promise<MagosRegistry | null> {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS)
  try {
    const res = await fetch(`${base}/admin/registry`, { signal: controller.signal })
    if (!res.ok) {
      console.warn(`[magos-plugin] /admin/registry returned ${res.status}; skipping model registration`)
      return null
    }
    return (await res.json()) as MagosRegistry
  } catch (err) {
    console.warn(`[magos-plugin] failed to reach ${base}/admin/registry: ${(err as Error).message}`)
    return null
  } finally {
    clearTimeout(timer)
  }
}

const plugin: Plugin = async () => ({
  // The config() hook fires before opencode reads cfg.provider, so it
  // can both default missing fields and inject the model dict that
  // opencode 1.14.33+ requires for non-models.dev providers. The
  // provider.models() hook is silently skipped for providers that
  // aren't in the models.dev catalog (provider.ts:1153 in opencode),
  // so injecting models here is the only path that registers them.
  config: async (input) => {
    const providers = (input.provider ??= {})
    const cfg = (providers.magos ??= {})
    const base = baseURL(cfg)
    cfg.name ??= "Magos"
    cfg.npm ??= "@ai-sdk/openai-compatible"
    // Default options.baseURL so a bare opencode.json works. The /v1 is
    // what the SDK appends paths to (POST /v1/chat/completions).
    const options = (cfg.options ??= {})
    options.baseURL ??= `${base}/v1`

    const registry = await fetchRegistry(base)
    if (!registry) return

    const models = (cfg.models ??= {})
    for (const entry of registry.entries) {
      if (entry.deprecated_at) continue
      const id = namespacedId(entry)
      models[id] ??= entryToModel(entry)
    }
  },
})

export default plugin
