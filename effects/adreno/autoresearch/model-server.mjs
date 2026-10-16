import { spawn } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { resolve, dirname } from 'node:path';
// Reuse private local authentication; never copy credentials into this repository.
const credentials = process.env.OPENCODE_PRIVATE_SETTINGS;
if (!credentials) throw new Error('OPENCODE_PRIVATE_SETTINGS must name the private local authentication file');
const settings = JSON.parse(await readFile(credentials, 'utf8'));
const root = resolve(dirname(fileURLToPath(import.meta.url)), '../../..');
const child = spawn(process.env.OPENCODE_BIN ?? 'opencode', ['serve', '--hostname', '127.0.0.1', '--port', '4321'], {
  cwd: root,
  env: { ...process.env, OPENCODE_SERVER_USERNAME: settings.username, OPENCODE_SERVER_PASSWORD: settings.password,
    OPENCODE_CONFIG_CONTENT: JSON.stringify({ model: 'opencode/deepseek-v4.1-flash-free', small_model: 'opencode/deepseek-v4.1-flash-free', permission: { '*': 'deny' }, autoupdate: false, share: 'disabled', compaction: { auto: false } }) },
  stdio: 'inherit',
});
const lifetime = setTimeout(() => child.kill('SIGTERM'), 12 * 3600 * 1000);
for (const signal of ['SIGTERM', 'SIGINT']) process.once(signal, () => child.kill(signal));
child.once('error', error => { clearTimeout(lifetime); console.error(error.message); process.exitCode = 1; });
child.once('close', code => { clearTimeout(lifetime); process.exitCode = code ?? 1; });
