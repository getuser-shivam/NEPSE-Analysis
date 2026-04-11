const https = require('https');
https.get({
  hostname: 'api.github.com',
  path: '/repos/getuser-shivam/NEPSE-Analysis/actions/runs/24274515117/jobs',
  headers: {'User-Agent': 'Node.js'}
}, res => {
  let data = '';
  res.on('data', chunk => data += chunk);
  res.on('end', () => {
    try {
      const parsed = JSON.parse(data);
      const fails = parsed.jobs.filter(j => j.conclusion === 'failure');
      if (fails.length > 0) {
         console.log(fails[0].url);
      }
    } catch(e) { }
  });
}).on('error', console.error);
