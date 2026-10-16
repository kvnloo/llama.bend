import { readFile, writeFile, mkdir, rename } from 'node:fs/promises';
import { resolve } from 'node:path';
const [directory, contextPath] = process.argv.slice(2);
if (!directory || !contextPath || !process.env.OPENCODE_PRIVATE_SETTINGS) throw new Error('usage: models.mjs <receipt-directory> <context.json>; set OPENCODE_PRIVATE_SETTINGS');
const settings = JSON.parse(await readFile(process.env.OPENCODE_PRIVATE_SETTINGS, 'utf8'));
const headers = { 'Content-Type': 'application/json', Authorization: 'Basic ' + Buffer.from(settings.username + ':' + settings.password).toString('base64') };
const base = 'http://127.0.0.1:4321';
const dir = resolve(directory);
await mkdir(dir, { recursive: true, mode: 0o700 });
const lifetime = new AbortController();
const running = new Set();
for (const signal of ['SIGTERM', 'SIGINT']) process.once(signal, () => lifetime.abort(new Error(signal)));
async function save(name, value) {
  const path = dir + '/' + name;
  const temporary = path + '.' + process.pid + '.tmp';
  await writeFile(temporary, JSON.stringify(value, null, 2) + '\n', { mode: 0o600 });
  await rename(temporary, path);
}
async function request(method, path, body, timeout = 180000, abortable = true) {
  const signal = abortable ? AbortSignal.any([lifetime.signal, AbortSignal.timeout(timeout)]) : AbortSignal.timeout(timeout);
  const reply = await fetch(base + path, { method, headers, body: body === undefined ? undefined : JSON.stringify(body), signal });
  if (!reply.ok) throw new Error('OpenCode HTTP ' + reply.status + ': ' + (await reply.text()).slice(0, 500));
  if (reply.status === 204) return null;
  return reply.json();
}
const providers = await request('GET', '/provider', undefined, 30000);
const provider = providers.all?.find(value => value.id === 'opencode');
if (!provider) throw new Error('OpenCode free provider unavailable; no fallback');
const models = Object.values(provider.models ?? {}).filter(model => model.cost?.input === 0 && model.cost?.output === 0 && model.cost?.cache?.read === 0 && model.cost?.cache?.write === 0 && (model.id.endsWith('-free') || model.id === 'big-pickle'));
const catalog = models.map(model => ({ id: model.id, name: model.name, cost: model.cost, limit: model.limit }));
await save('catalog.json', { observed_at: new Date().toISOString(), provider: 'opencode', models: catalog, no_paid_fallback: true });
console.log('FREE_MODEL_CATALOG ' + JSON.stringify(catalog));
if (!models.some(model => model.id === 'deepseek-v4.1-flash-free')) throw new Error('Required free DeepSeek model unavailable');
const context = JSON.parse(await readFile(contextPath, 'utf8'));
if (context.discover_only) process.exit(0);
const focus = [
  'Coalesced global-memory loads and exact four-accumulator cooperative arithmetic.',
  'Occupancy, register pressure and local-size selection on Adreno wavefronts.',
  'Vectorized float4 rowwise layout versus scalar and workgroup overhead.',
  'Local-memory reduction trees, barrier cost and correct multi-row indexing.',
  'Loop unrolling choices, compiler heuristics and avoiding register spills.',
  'Robust paired measurements, thermal/DVFS drift and falsifiable explanations.',
];
async function work(model, index) {
  const filename = model.id + '.json';
  let record;
  try { record = JSON.parse(await readFile(dir + '/' + filename, 'utf8')); }
  catch (error) { if (error.code !== 'ENOENT') throw error; }
  if (record) {
    // Do not duplicate a request whose response may already have reached the server.
    console.log('FREE_MODEL_RECOVERED ' + model.id + ' ' + record.status);
    return record;
  }
  record = { model: model.id, provider: 'opencode', status: 'starting', started_at: new Date().toISOString() };
  await save(filename, record);
  try {
    const session = await request('POST', '/session', { title: 'Adreno autoresearch: ' + model.id });
    record.session_id = session.id;
    record.status = 'running';
    await save(filename, record);
    running.add(session.id);
    console.log('FREE_MODEL_STARTED ' + model.id + ' ' + session.id);
    const prompt = `Research a measured kernel optimization, not an imagined benchmark. ${focus[index % focus.length]}\n\nWe optimize one real 576x576 row-major F32 SmolLM2 query projection through a Bend foreign-C effect on Adreno830, OpenCL3.0 Android. It is NOT native Bend ! or whole-model inference. Candidate kernels must work for arbitrary finite x, and every row of five nontrivial inputs is checked against production Bend M.mv with fixed atol2e-5+rtol2e-4*abs(cpu). No relaxed math or contraction. Physical measurements use profiling events, paired serialized controls and thermal limits. You may suggest ONLY parameter combinations accepted by the attached schema, never arbitrary source/commands. The controller generates vetted OpenCL; your suggestions cannot alter acceptance or runtime limits. Learn from actual observations; do not fabricate speedups. Return one JSON object, no markdown: {\"candidates\":[{spec1},{spec2}],\"reason\":\"concise evidence/tradeoff\"}, at most 6 distinct candidates and 1200 words. Context:\n${JSON.stringify(context)}`;
    const reply = await request('POST', '/session/' + session.id + '/message', {
      model: { providerID: 'opencode', modelID: model.id },
      tools: { bash: false, edit: false, write: false, apply_patch: false, read: false, glob: false, grep: false, task: false, webfetch: false, websearch: false },
      parts: [{ type: 'text', text: prompt }],
    });
    const report = (reply.parts ?? []).filter(part => part.type === 'text').map(part => part.text).join('\n\n');
    const info = reply.info ?? {};
    record = { ...record, report, usage: { tokens: info.tokens, cost: info.cost, finish: info.finish, error: info.error }, status: !info.error && ['stop', 'end_turn'].includes(info.finish) && report.trim() ? 'completed' : 'incomplete' };
    if (typeof info.cost === 'number' && info.cost !== 0) throw new Error('BILLING_GUARD: nonzero cost reported for advertised free model');
    if (record.status === 'completed') {
      try {
        const clean = report.trim().replace(/^```(?:json)?\s*/, '').replace(/\s*```$/, '');
        const proposals = JSON.parse(clean);
        if (!Array.isArray(proposals.candidates) || proposals.candidates.length > 6) throw new Error('Expected at most six specs');
        record.candidates = proposals.candidates;
        record.reason = String(proposals.reason ?? '').slice(0, 12000);
      } catch (error) { record.status = 'invalid-proposal'; record.parse_error = error.message; }
    }
  } catch (error) {
    record.status = 'incomplete';
    record.error = String(error.message).split(settings.password).join('[REDACTED]');
    if (record.session_id) {
      try { await request('POST', '/session/' + record.session_id + '/abort', undefined, 15000, false); record.abort_requested = true; }
      catch { record.abort_requested = false; }
    }
  } finally {
    if (record.session_id) running.delete(record.session_id);
    record.finished_at = new Date().toISOString();
    await save(filename, record);
    console.log('FREE_MODEL_FINISHED ' + model.id + ' ' + record.status);
  }
  return record;
}
// One concurrent request per discovered free model; no paid aliases or recursive agents.
const results = await Promise.all(models.map(work));
await save('round.json', { finished_at: new Date().toISOString(), models: results.map(value => ({ model: value.model, status: value.status, session_id: value.session_id })), completed: results.filter(value => value.status === 'completed').length });
if (results.some(value => value.error?.includes('BILLING_GUARD'))) process.exitCode = 2;
