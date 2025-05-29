import React from 'react';
import { Table, TableBody, TableCell, TableHead, TableRow } from '@mui/material';
import { SessionRow } from './SessionRow';

export const SessionTable = ({ sessions, onPause, onResume, onStop }) => (
  <Table>
    <TableHead>
      <TableRow>
        <TableCell>Session ID</TableCell>
        <TableCell>Status</TableCell>
        <TableCell>Start Time</TableCell>
        <TableCell>Symbols</TableCell>
        <TableCell>Controls</TableCell>
      </TableRow>
    </TableHead>
    <TableBody>
      {sessions.map(session => (
        <SessionRow
          key={session.session_id}
          session={session}
          onPause={onPause}
          onResume={onResume}
          onStop={onStop}
        />
      ))}
    </TableBody>
  </Table>
);
