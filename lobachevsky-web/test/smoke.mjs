// Смоук-тест: поднимает статический сервер, открывает все миры и проверяет, что нет ошибок.
// Запуск из корня проекта: node test/smoke.mjs   (нужен пакет playwright)
import { createRequire } from 'node:module';
import http from 'node:http';
import { readFile, mkdir } from 'node:fs/promises';
import { extname, join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const { chromium } = createRequire(import.meta.url)('playwright');
const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const types = { '.html':'text/html', '.js':'text/javascript', '.css':'text/css', '.json':'application/json', '.glsl':'text/plain', '.vert':'text/plain' };
const server = http.createServer(async (req, res) => {
  const p = join(root, decodeURIComponent(new URL(req.url, 'http://x').pathname).replace(/\/$/, '/index.html'));
  try { res.writeHead(200, { 'content-type': types[extname(p)] || 'application/octet-stream' }); res.end(await readFile(p)); }
  catch { res.writeHead(404); res.end(); }
}).listen(0);
const port = server.address().port;

const browser = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
const page = await browser.newPage({ viewport: { width: 480, height: 300 } });
const errors = [];
page.on('pageerror', e => errors.push(e.message));
page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
await page.goto(`http://localhost:${port}/index.html`);
await page.waitForFunction(() => window.__lob && window.__lob.app.world, null, { timeout: 30000 });
await mkdir(join(root, 'test/shots'), { recursive: true });
const n = await page.evaluate(() => document.querySelectorAll('#worlds button').length);
let ok = true;
for (let i = 1; i <= n; i++) {
  await page.keyboard.press('Digit' + i);
  await page.waitForTimeout(1500);
  const id = await page.evaluate(() => window.__lob.app.world.id);
  await page.screenshot({ path: join(root, `test/shots/${i}-${id}.png`), timeout: 120000 });
  console.log(`мир ${i}: ${id}`);
}
if (errors.length) { ok = false; console.error('Ошибки:\n' + errors.join('\n')); }
await browser.close(); server.close();
console.log(ok ? 'Все миры открылись без ошибок.' : 'Есть ошибки.');
process.exit(ok ? 0 : 1);
