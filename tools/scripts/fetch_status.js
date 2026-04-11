const https = require('https');
https.get({
  hostname: 'api.github.com',
  path: '/repos/getuser-shivam/NEPSE-Analysis/actions/runs?per_page=10',
  headers: {'User-Agent': 'Node.js'}
}, res => {
  let data = '';
  res.on('data', chunk => data += chunk);
  res.on('end', () => {
    try {
      const parsed = JSON.parse(data);
      const runs = parsed.workflow_runs.filter(r => r.name === 'NEPSE Analysis CI/CD');
      if (runs.length === 0) return console.log('No main runs found in top 10');
      
      const latestRunId = runs[0].id;
      console.log(`Latest Main Run ID: ${latestRunId}`);
      console.log(`URL: ${runs[0].html_url}`);
      
      https.get({
        hostname: 'api.github.com',
        path: `/repos/getuser-shivam/NEPSE-Analysis/actions/runs/${latestRunId}/jobs`,
        headers: {'User-Agent': 'Node.js'}
      }, res2 => {
        let data2 = '';
        res2.on('data', chunk => data2 += chunk);
        res2.on('end', () => {
          const parsed2 = JSON.parse(data2);
          parsed2.jobs.forEach(j => {
            console.log(`${j.name}: ${j.status} | ${j.conclusion}`);
            if (j.conclusion === 'failure') {
               j.steps.forEach(s => {
                 if (s.conclusion === 'failure') console.log(`  Failed step: ${s.name} (${s.number})`);
               });
            }
          });
        });
      }).on('error', console.error);
    } catch(e) { }
  });
}).on('error', console.error);
