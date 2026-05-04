import type { PluginModule } from "@opencode-ai/plugin"
import type { Model as ModelV2, Provider as ProviderV2 } from "@opencode-ai/sdk/v2"

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

/**
 * Resolve the magos host root.
 *
 * Priority: opencode.jsonc `options.baseURL` > `MAGOS_BASE_URL` env var >
 * `DEFAULT_BASE_URL`. The opencode value typically has a `/v1` suffix
 * (the SDK uses it verbatim for chat completions); we strip it so the
 * same root works for `/admin/registry`.
 */
function baseURL(provider?: ProviderV2): string {
  const fromOptions = provider?.options?.baseURL
  const raw =
    (typeof fromOptions === "string" && fromOptions) || process.env.MAGOS_BASE_URL || DEFAULT_BASE_URL
  return raw.replace(/\/+$/, "").replace(/\/v1$/, "")
}

function namespacedId(entry: MagosEntry): string {
  return `${entry.provider}/${entry.raw_id}`
}

function modalityFlags(modalities: string[]) {
  const has = (m: string) => modalities.includes(m)
  return {
    text: has("text"),
    audio: has("audio"),
    image: has("image"),
    video: has("video"),
    // Magos emits `file` for arbitrary file uploads (PDFs, etc.); OpenCode
    // tracks pdf specifically. Map file -> pdf as the closest fit.
    pdf: has("file"),
  }
}

function entryToModel(entry: MagosEntry, base: string): ModelV2 {
  const input = modalityFlags(entry.input_modalities)
  const output = modalityFlags(entry.output_modalities)
  const attachment = entry.input_modalities.some((m) => m !== "text")
  const id = namespacedId(entry)
  // api.id is the wire-level `model` field that the bundled
  // @ai-sdk/openai-compatible adapter sends in the request body.
  // Must match the namespaced model id, not the provider id, or magos
  // sees `model: "magos"` and no rule matches.
  return {
    id,
    providerID: "magos",
    api: {
      id,
      url: `${base}/v1`,
      npm: "@ai-sdk/openai-compatible",
    },
    name: id,
    capabilities: {
      temperature: true,
      reasoning: false,
      attachment,
      toolcall: true,
      input,
      output,
      interleaved: false,
    },
    cost: {
      input: entry.input_cost ?? 0,
      output: entry.output_cost ?? 0,
      cache: {
        read: entry.cache_read_cost ?? entry.input_cost ?? 0,
        write: entry.cache_write_cost ?? 0,
      },
    },
    limit: {
      context: entry.context_size ?? 0,
      output: entry.max_output ?? entry.context_size ?? 0,
    },
    status: entry.deprecated_at ? "deprecated" : "active",
    options: {},
    headers: {},
    release_date: "",
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

async function fetchModels(provider: ProviderV2): Promise<Record<string, ModelV2>> {
  const base = baseURL(provider)
  const registry = await fetchRegistry(base)
  if (!registry) return {}
  const models: Record<string, ModelV2> = {}
  for (const entry of registry.entries) {
    if (entry.deprecated_at) continue
    const model = entryToModel(entry, base)
    models[model.id] = model
  }
  return models
}

const plugin: PluginModule = {
  id: "magos",
  server: async () => ({
    // Inject default `provider.magos` fields so users don't have to
    // declare them in opencode.json. Field-level merge: anything the
    // user supplied wins, only missing fields get defaulted.
    config: async (input) => {
      const providers = (input.provider ??= {})
      const cfg = (providers.magos ??= {})
      cfg.name ??= "Magos"
    },
    provider: {
      id: "magos",
      models: async (provider) => fetchModels(provider),
    },
  }),
}

export default plugin
