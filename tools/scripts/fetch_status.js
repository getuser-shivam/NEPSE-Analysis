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
      const runs = parsed.workflow_runs.filter(r => r.name === 'Deploy Pages');
      if (runs.length === 0) {
        console.log('No Deploy Pages runs found yet. Listing all runs:');
        parsed.workflow_runs.slice(0, 5).forEach(r => {
          console.log(`  ${r.name}: ${r.status} | ${r.conclusion} | ${r.html_url}`);
        });
        return;
      }
      const run = runs[0];
      console.log(`Deploy Pages Run: ${run.id}`);
      console.log(`Status: ${run.status} | Conclusion: ${run.conclusion}`);
      console.log(`URL: ${run.html_url}`);
      
      if (run.status === 'completed') {
        https.get({
          hostname: 'api.github.com',
          path: `/repos/getuser-shivam/NEPSE-Analysis/actions/runs/${run.id}/jobs`,
          headers: {'User-Agent': 'Node.js'}
        }, res2 => {
          let data2 = '';
          res2.on('data', c => data2 += c);
          res2.on('end', () => {
            const p2 = JSON.parse(data2);
            p2.jobs.forEach(j => {
              console.log(`  Job: ${j.name} | ${j.conclusion}`);
              if (j.conclusion === 'failure') {
                j.steps.forEach(s => {
                  if (s.conclusion === 'failure') console.log(`    FAILED: ${s.name}`);
                });
              }
            });
          });
        });
      }
    } catch(e) { console.error(e); }
  });
});
