import React, { useState } from 'react';
import { 
  Button, 
  Dialog, 
  DialogTitle, 
  DialogContent, 
  DialogActions,
  Typography,
  Box,
  CircularProgress,
  Stepper,
  Step,
  StepLabel,
  Alert,
  Chip
} from '@mui/material';
import RocketLaunchIcon from '@mui/icons-material/RocketLaunch';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import WarningIcon from '@mui/icons-material/Warning';
import { usePaperTrading } from '../../../context/PaperTradingContext';
import { useNotification } from '../../../components/common/NotificationSystem';

// Pre-flight check steps
const PREFLIGHT_STEPS = [
  'Data Source Connectivity',
  'Strategy Readiness',
  'Risk Parameters',
  'System Resources'
];

// Initialization steps
const INITIALIZATION_STEPS = [
  'Data Synchronization',
  'Strategy Warm-up',
  'Portfolio Analysis',
  'Trading Activation'
];

interface AutonomousTradingButtonProps {
  configPath: string;
  duration: number;
  interval: number;
}

const AutonomousTradingButton: React.FC<AutonomousTradingButtonProps> = ({ 
  configPath, 
  duration, 
  interval 
}) => {
  const { startAutonomousTrading, state } = usePaperTrading();
  const { showNotification } = useNotification();
  
  // Dialog state
  const [open, setOpen] = useState(false);
  const [preflightComplete, setPreflightComplete] = useState(false);
  const [preflightActiveStep, setPreflightActiveStep] = useState(0);
  const [preflightErrors, setPreflightErrors] = useState<{[key: number]: string}>({});
  
  // Initialization state
  const [initializationStarted, setInitializationStarted] = useState(false);
  const [initializationActiveStep, setInitializationActiveStep] = useState(0);
  const [initializationComplete, setInitializationComplete] = useState(false);
  
  // Handle opening the dialog
  const handleOpen = () => {
    setOpen(true);
    // Reset states
    setPreflightComplete(false);
    setPreflightActiveStep(0);
    setPreflightErrors({});
    setInitializationStarted(false);
    setInitializationActiveStep(0);
    setInitializationComplete(false);
    
    // Start pre-flight checks
    runPreflightChecks();
  };
  
  // Handle closing the dialog
  const handleClose = () => {
    if (initializationStarted && !initializationComplete) {
      // Show confirmation dialog if initialization is in progress
      if (window.confirm('Canceling now will abort the initialization process. Are you sure?')) {
        setOpen(false);
      }
    } else {
      setOpen(false);
    }
  };
  
  // Run pre-flight checks
  const runPreflightChecks = async () => {
    const errors: {[key: number]: string} = {};
    
    // Simulate pre-flight checks with delays
    for (let i = 0; i < PREFLIGHT_STEPS.length; i++) {
      setPreflightActiveStep(i);
      
      // Simulate check with delay
      await new Promise(resolve => setTimeout(resolve, 800));
      
      // Simulate random errors (in a real implementation, these would be actual checks)
      if (PREFLIGHT_STEPS[i] === 'Data Source Connectivity' && Math.random() > 0.8) {
        errors[i] = 'Unable to connect to data source. Please check your network connection.';
      } else if (PREFLIGHT_STEPS[i] === 'Strategy Readiness' && Math.random() > 0.9) {
        errors[i] = 'Strategy parameters are not properly configured.';
      }
    }
    
    // Update errors state
    setPreflightErrors(errors);
    
    // Check if all pre-flight checks passed
    if (Object.keys(errors).length === 0) {
      setPreflightComplete(true);
    }
  };
  
  // Start initialization process
  const startInitialization = async () => {
    setInitializationStarted(true);
    
    // Simulate initialization steps with delays
    for (let i = 0; i < INITIALIZATION_STEPS.length; i++) {
      setInitializationActiveStep(i);
      
      // Simulate step with delay
      await new Promise(resolve => setTimeout(resolve, 1200));
    }
    
    // Complete initialization
    setInitializationComplete(true);
    
    // Start autonomous trading
    try {
      await startAutonomousTrading({
        configPath,
        duration,
        interval,
        autonomousMode: true
      });
      
      showNotification({
        type: 'success',
        title: 'Autonomous Trading Started',
        message: 'The trading agent is now running in fully autonomous mode'
      });
      
      // Close dialog after a short delay
      setTimeout(() => {
        setOpen(false);
      }, 1500);
    } catch (error) {
      showNotification({
        type: 'error',
        title: 'Failed to Start Autonomous Trading',
        message: error instanceof Error ? error.message : 'Unknown error occurred'
      });
    }
  };
  
  // Render step icon based on status
  const getStepIcon = (index: number, activeStep: number, errors: {[key: number]: string}) => {
    if (index < activeStep) {
      return errors[index] ? <ErrorIcon color="error" /> : <CheckCircleIcon color="success" />;
    } else if (index === activeStep) {
      return <CircularProgress size={20} />;
    }
    return null;
  };
  
  return (
    <>
      <Button
        variant="contained"
        color="primary"
        startIcon={<RocketLaunchIcon />}
        onClick={handleOpen}
        disabled={state.isLoading}
        sx={{ 
          bgcolor: 'primary.main',
          '&:hover': {
            bgcolor: 'primary.dark',
          },
          fontWeight: 'bold'
        }}
      >
        Start Autonomous Trading
      </Button>
      
      <Dialog
        open={open}
        onClose={handleClose}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          <Box display="flex" alignItems="center">
            <RocketLaunchIcon sx={{ mr: 1 }} />
            <Typography variant="h6">
              Autonomous Trading Initialization
            </Typography>
          </Box>
        </DialogTitle>
        
        <DialogContent>
          {/* Pre-flight Checks */}
          <Box mb={4}>
            <Typography variant="h6" gutterBottom>
              Pre-flight Checks
              {preflightComplete && (
                <Chip 
                  label="Complete" 
                  color="success" 
                  size="small" 
                  sx={{ ml: 2 }} 
                />
              )}
            </Typography>
            
            <Stepper activeStep={preflightActiveStep} orientation="vertical">
              {PREFLIGHT_STEPS.map((label, index) => (
                <Step key={label}>
                  <StepLabel
                    StepIconComponent={() => 
                      getStepIcon(index, preflightActiveStep, preflightErrors)
                    }
                  >
                    {label}
                    {preflightErrors[index] && (
                      <Alert severity="error" sx={{ mt: 1 }}>
                        {preflightErrors[index]}
                      </Alert>
                    )}
                  </StepLabel>
                </Step>
              ))}
            </Stepper>
          </Box>
          
          {/* Initialization Process */}
          {preflightComplete && (
            <Box>
              <Typography variant="h6" gutterBottom>
                Initialization Process
                {initializationComplete && (
                  <Chip 
                    label="Complete" 
                    color="success" 
                    size="small" 
                    sx={{ ml: 2 }} 
                  />
                )}
              </Typography>
              
              {!initializationStarted ? (
                <Alert severity="info" sx={{ mb: 2 }}>
                  All pre-flight checks passed. Ready to start initialization.
                </Alert>
              ) : (
                <Stepper activeStep={initializationActiveStep} orientation="vertical">
                  {INITIALIZATION_STEPS.map((label, index) => (
                    <Step key={label}>
                      <StepLabel
                        StepIconComponent={() => 
                          getStepIcon(index, initializationActiveStep, {})
                        }
                      >
                        {label}
                      </StepLabel>
                    </Step>
                  ))}
                </Stepper>
              )}
            </Box>
          )}
          
          {/* Error Summary */}
          {Object.keys(preflightErrors).length > 0 && (
            <Alert severity="warning" sx={{ mt: 2 }}>
              <Typography variant="subtitle1">
                Cannot proceed with initialization due to the following issues:
              </Typography>
              <ul>
                {Object.entries(preflightErrors).map(([step, error]) => (
                  <li key={step}>
                    <strong>{PREFLIGHT_STEPS[parseInt(step)]}</strong>: {error}
                  </li>
                ))}
              </ul>
            </Alert>
          )}
          
          {/* Success Message */}
          {initializationComplete && (
            <Alert severity="success" sx={{ mt: 2 }}>
              <Typography variant="subtitle1">
                Autonomous trading has been successfully initialized!
              </Typography>
              <Typography variant="body2">
                The trading agent is now running in fully autonomous mode and will make trading decisions based on the configured strategy.
              </Typography>
            </Alert>
          )}
        </DialogContent>
        
        <DialogActions>
          {preflightComplete && !initializationStarted && (
            <Button 
              onClick={startInitialization} 
              color="primary" 
              variant="contained"
              startIcon={<RocketLaunchIcon />}
            >
              Start Initialization
            </Button>
          )}
          
          <Button 
            onClick={handleClose} 
            color={initializationComplete ? "primary" : "secondary"}
          >
            {initializationComplete ? "Close" : "Cancel"}
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default AutonomousTradingButton;
