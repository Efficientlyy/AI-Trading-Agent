import React from 'react';
import { Button, Stack } from '@mui/material';

interface Props {
  onPauseAll: () => void;
  onResumeAll: () => void;
  onStopAll: () => void;
  systemStatus: string;
}

export const SystemControlPanel: React.FC<Props> = ({ onPauseAll, onResumeAll, onStopAll, systemStatus }) => (
  <Stack direction="row" spacing={2} alignItems="center">
    <Button variant="contained" color="warning" onClick={onPauseAll}>Pause All</Button>
    <Button variant="contained" color="success" onClick={onResumeAll}>Resume All</Button>
    <Button variant="contained" color="error" onClick={onStopAll}>Stop All</Button>
    <span style={{ marginLeft: 16, fontWeight: 'bold' }}>System Status: {systemStatus}</span>
  </Stack>
);
