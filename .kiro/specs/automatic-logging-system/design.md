# Design Document

## Overview

The automatic logging system provides a seamless way to capture all command executions and outputs in Python scripts through a single function call. The system builds on existing logging infrastructure while enhancing it to meet the specific requirement of using `start_logging()` as the entry point and providing more comprehensive command tracking.

## Architecture

The system follows a stream interception architecture that captures output at the system level while preserving normal script execution. The design consists of three main layers:

1. **Interface Layer**: Simple `start_logging()` function that serves as the single entry point
2. **Capture Layer**: Enhanced stream capture that intercepts stdout/stderr and function calls
3. **Logging Layer**: Structured logging with timestamps and clear formatting

```mermaid
graph TD
    A[start_logging()] --> B[Enhanced Stream Capture]
    B --> C[Command Tracker]
    B --> D[Output Interceptor]
    C --> E[Log Formatter]
    D --> E
    E --> F[File Logger]
    E --> G[Console Passthrough]
```

## Components and Interfaces

### 1. Main Interface Function

```python
def start_logging(log_file: str = "", log_dir: str = "") -> Logger:
    """
    Single entry point for automatic logging.
    
    Args:
        log_file: Name of log file (defaults to script_name.log)
        log_dir: Directory for logs (defaults to output/logs)
    
    Returns:
        Logger instance for optional manual logging
    """
```

### 2. Enhanced Stream Capture

Extends the existing `StreamCapture` class to provide:
- Better command detection and logging
- Improved output filtering and formatting
- Function call tracking capabilities
- Error handling and recovery

### 3. Command Tracker

New component that tracks:
- Function calls with parameters
- Variable assignments
- Import statements
- Control flow operations

### 4. Log Formatter

Enhanced formatting that provides:
- Clear command/output distinction
- Timestamps for all entries
- Hierarchical structure for nested operations
- Error context and stack traces

## Data Models

### LogEntry Structure
```python
@dataclass
class LogEntry:
    timestamp: datetime
    entry_type: str  # 'COMMAND', 'OUTPUT', 'ERROR'
    content: str
    context: Optional[str] = None
    level: int = 0  # For nested operations
```

### Logger Configuration
```python
@dataclass
class LoggerConfig:
    log_file: str
    log_dir: str
    capture_commands: bool = True
    capture_outputs: bool = True
    include_timestamps: bool = True
    filter_noise: bool = True
```

## Error Handling

The system implements robust error handling:

1. **Logging Failures**: If log file writing fails, continue script execution and attempt to log to a fallback location
2. **Stream Restoration**: Ensure original stdout/stderr are restored if capture fails
3. **Performance Issues**: Implement buffering and async writing to prevent blocking
4. **Memory Management**: Automatic cleanup of large log buffers

## Testing Strategy

### Unit Tests
- Test stream capture functionality
- Test log formatting and output
- Test error handling scenarios
- Test performance with large outputs

### Integration Tests
- Test with real Python scripts
- Test with different types of operations (loops, functions, imports)
- Test log file creation and writing
- Test concurrent logging scenarios

### Performance Tests
- Measure overhead of logging system
- Test with high-volume output scripts
- Memory usage profiling
- File I/O performance testing

## Implementation Approach

### Phase 1: Core Interface
- Implement `start_logging()` function as wrapper around existing `setup_logging()`
- Add default parameter handling
- Ensure backward compatibility

### Phase 2: Enhanced Capture
- Extend `StreamCapture` with command detection
- Implement better output filtering
- Add function call tracking

### Phase 3: Advanced Features
- Add command context tracking
- Implement nested operation logging
- Add performance optimizations

### Phase 4: Testing and Refinement
- Comprehensive test suite
- Performance optimization
- Documentation and examples

## Key Design Decisions

1. **Stream Interception vs Code Instrumentation**: Using stream interception to avoid requiring code modifications
2. **Single Entry Point**: `start_logging()` provides the simple interface requested
3. **Backward Compatibility**: Build on existing `setup_logging()` infrastructure
4. **Performance First**: Minimize overhead through efficient buffering and filtering
5. **Fail-Safe Operation**: Never crash the main script due to logging issues