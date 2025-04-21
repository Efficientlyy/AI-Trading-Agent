try {
  require('react-router-dom');
  console.log('SUCCESS');
} catch (e) {
  console.error('FAIL:', e.message);
}
