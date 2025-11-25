// Generates docs/wumbo-engine.pdf from docs/wumbo-engine.html via Puppeteer
const fs = require('fs');
const path = require('path');
const puppeteer = require('puppeteer');

(async () => {
  const root = process.cwd();
  const htmlPath = path.join(root, 'docs', 'wumbo-engine.html');
  const outPath = path.join(root, 'docs', 'wumbo-engine.pdf');
  if (!fs.existsSync(htmlPath)) {
    console.error('Missing docs/wumbo-engine.html');
    process.exit(1);
  }

  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });
  try {
    const page = await browser.newPage();
    await page.goto('file://' + htmlPath, { waitUntil: 'networkidle0' });
    await page.emulateMediaType('screen');
    await page.pdf({
      path: outPath,
      format: 'A4',
      printBackground: true,
      margin: { top: '14mm', right: '12mm', bottom: '16mm', left: '12mm' }
    });
    console.log('Wrote', outPath);
  } finally {
    await browser.close();
  }
})();

