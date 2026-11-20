const fs = require('fs'), path = require('path'), S = p => fs.readFileSync(path.join(__dirname, 'src', p), 'utf8');
const recipes = {}; for (const f of fs.readdirSync(path.join(__dirname, 'src/recipes'))) recipes[f.replace('.js', '')] = S('recipes/' + f);
const app = ['core.js', 'tools.js', 'ui_a.js', 'ui_b.js'].map(S);
const js = 'const RECIPES = ' + JSON.stringify(recipes) + ';\n' + app.join('\n').replace(/<\/script/gi, '<\\/script');
const tf = fs.readFileSync(path.join(__dirname, 'node_modules/@tensorflow/tfjs/dist/tf.min.js'), 'utf8').replace(/<\/script/gi, '<\\/script').replace(/\/\/# sourceMappingURL=.*/g, '');
let html = S('index.html').replace('/*CSS*/', () => S('style.css')).replace('/*TF*/', () => tf).replace('/*APP*/', () => js);
fs.mkdirSync(path.join(__dirname, 'dist'), { recursive: true }); fs.writeFileSync(path.join(__dirname, 'dist/genlab.html'), html);
console.log('built', (html.length / 1e6).toFixed(2), 'MB');
