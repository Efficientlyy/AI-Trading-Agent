import React from 'react';
import { TableRow, TableCell, Button } from '@mui/material';

export const SessionRow = ({ session, onPause, onResume, onStop }) => (
  <TableRow>
    <TableCell>{session.session_id}</TableCell>
    <TableCell>{session.status}</TableCell>
    <TableCell>{session.start_time}</TableCell>
    <TableCell>{(session.symbols || []).join(', ')}</TableCell>
    <TableCell>
      {session.status === 'running' && (
        <Button color="warning" onClick={() => onPause(session.session_id)}>Pause</Button>
      )}
      {session.status === 'paused' && (
        <Button color="success" onClick={() => onResume(session.session_id)}>Resume</Button>
      )}
      {['running', 'paused', 'starting'].includes(session.status) && (
        <Button color="error" onClick={() => onStop(session.session_id)}>Stop</Button>
      )}
    </TableCell>
  </TableRow>
);
