"""
Automatic logging utilities for Python scripts.

This module provides simple automatic logging that captures:
- All print statements and console output
- Basic script operations

Usage:
    from utils.logging_utils import start_logging
    
    # Simple setup - handles everything automatically
    logger = start_logging(log_file='my_script.log')
    
    # Everything is now logged automatically:
    print("This goes to console and log")
    # No manual logging needed!
"""
import os
import sys
import logging
import inspect
import re
import traceback
import ast
from pathlib import Path


class CodeBlockAnalyzer:
    """
    Analyzes Python source code to identify logical code blocks.
    """
    def __init__(self, source_code):
        self.source_code = source_code
        self.source_lines = source_code.splitlines()
        self.code_blocks = {}  # Map line numbers to code blocks
        
    def analyze_blocks(self):
        """Analyze the source code to identify logical code blocks."""
        try:
            tree = ast.parse(self.source_code)
            self._visit_node(tree)
        except SyntaxError:
            # If we can't parse the AST, fall back to line-by-line analysis
            pass
            
    def _visit_node(self, node):
        """Visit AST nodes to identify code blocks."""
        if isinstance(node, (ast.For, ast.While)):
            # For/while loops - group the entire loop structure
            self._create_loop_block(node)
        elif isinstance(node, ast.If):
            # If statements - group the entire if/elif/else structure
            self._create_if_block(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Function definitions - group the entire function
            self._create_function_block(node)
        elif isinstance(node, ast.ClassDef):
            # Class definitions - group the entire class
            self._create_class_block(node)
        elif isinstance(node, (ast.Try, ast.With)):
            # Try/except and with statements - group the entire structure
            self._create_compound_block(node)
            
        # Continue visiting child nodes
        for child in ast.iter_child_nodes(node):
            self._visit_node(child)
            
    def _create_loop_block(self, node):
        """Create a code block for a loop structure."""
        start_line = node.lineno
        end_line = self._get_end_line(node)
        block_lines = self._get_block_lines(start_line, end_line)
        
        self.code_blocks[start_line] = {
            'type': 'loop',
            'start_line': start_line,
            'end_line': end_line,
            'code': '\n'.join(block_lines),
            'lines': list(range(start_line, end_line + 1))
        }
        
    def _create_if_block(self, node):
        """Create a code block for an if/elif/else structure."""
        start_line = node.lineno
        end_line = self._get_end_line(node)
        block_lines = self._get_block_lines(start_line, end_line)
        
        self.code_blocks[start_line] = {
            'type': 'conditional',
            'start_line': start_line,
            'end_line': end_line,
            'code': '\n'.join(block_lines),
            'lines': list(range(start_line, end_line + 1))
        }
        
    def _create_function_block(self, node):
        """Create a code block for a function definition."""
        start_line = node.lineno
        end_line = self._get_end_line(node)
        block_lines = self._get_block_lines(start_line, end_line)
        
        self.code_blocks[start_line] = {
            'type': 'function',
            'start_line': start_line,
            'end_line': end_line,
            'code': '\n'.join(block_lines),
            'lines': list(range(start_line, end_line + 1))
        }
        
    def _create_class_block(self, node):
        """Create a code block for a class definition."""
        start_line = node.lineno
        end_line = self._get_end_line(node)
        block_lines = self._get_block_lines(start_line, end_line)
        
        self.code_blocks[start_line] = {
            'type': 'class',
            'start_line': start_line,
            'end_line': end_line,
            'code': '\n'.join(block_lines),
            'lines': list(range(start_line, end_line + 1))
        }
        
    def _create_compound_block(self, node):
        """Create a code block for compound statements like try/except, with."""
        start_line = node.lineno
        end_line = self._get_end_line(node)
        block_lines = self._get_block_lines(start_line, end_line)
        
        self.code_blocks[start_line] = {
            'type': 'compound',
            'start_line': start_line,
            'end_line': end_line,
            'code': '\n'.join(block_lines),
            'lines': list(range(start_line, end_line + 1))
        }
        
    def _get_end_line(self, node):
        """Get the end line of an AST node."""
        if hasattr(node, 'end_lineno') and node.end_lineno:
            return node.end_lineno
        
        # Fallback: find the last line by examining child nodes
        max_line = node.lineno
        for child in ast.walk(node):
            if hasattr(child, 'lineno') and child.lineno > max_line:
                max_line = child.lineno
        return max_line
        
    def _get_block_lines(self, start_line, end_line):
        """Get the source code lines for a block."""
        lines = []
        for i in range(start_line - 1, min(end_line, len(self.source_lines))):
            lines.append(self.source_lines[i])
        return lines
        
    def get_block_for_line(self, line_number):
        """Get the code block that contains the given line number."""
        for start_line, block in self.code_blocks.items():
            if line_number in block['lines']:
                return block
        return None


class CommandTracker:
    """
    Tracks command execution using a hybrid approach that combines source analysis with execution hooks.
    """
    def __init__(self, logger):
        self.logger = logger
        self.script_filename = None
        self.source_lines = []  # Cache source lines for performance
        self.meaningful_commands = {}  # Map line numbers to commands
        self.logged_lines = set()  # Track what we've already logged
        self.logged_blocks = set()  # Track which code blocks we've already logged
        self.block_analyzer = None  # Code block analyzer
        
    def set_script_filename(self, filename):
        """Set the main script filename to track."""
        self.script_filename = filename
        # Pre-load and analyze source lines
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                source_code = f.read()
                self.source_lines = source_code.splitlines()
            
            # Initialize block analyzer
            self.block_analyzer = CodeBlockAnalyzer(source_code)
            self.block_analyzer.analyze_blocks()
            
            self._analyze_commands()
        except (IOError, OSError):
            self.source_lines = []
        
    def _analyze_commands(self):
        """Analyze source code to identify meaningful commands."""
        for line_num, line in enumerate(self.source_lines, 1):
            if self._is_meaningful_line(line):
                self.meaningful_commands[line_num] = line.strip()
        
    def get_current_command(self):
        """Get the current command being executed by inspecting the call stack."""
        try:
            # Get the current frame
            frame = inspect.currentframe()
            if frame:
                # Walk up the stack to find the main script frame
                current_frame = frame
                while current_frame:
                    filename = current_frame.f_code.co_filename
                    if filename == self.script_filename:
                        line_number = current_frame.f_lineno
                        if line_number in self.meaningful_commands:
                            return line_number, self.meaningful_commands[line_number]
                    current_frame = current_frame.f_back
        except Exception:
            pass
        return None, None
        
    def log_current_command(self):
        """Log the current command if it's meaningful and hasn't been logged."""
        line_number, command = self.get_current_command()
        if line_number and line_number not in self.logged_lines:
            self.logger.info(f"[COMMAND:L{line_number}] {command}")
            self.logged_lines.add(line_number)
            return True
        return False
        
    def log_command_if_meaningful(self, line_number):
        """Log a command if it's meaningful and hasn't been logged yet."""
        if line_number in self.meaningful_commands and line_number not in self.logged_lines:
            command = self.meaningful_commands[line_number]
            self.logger.info(f"[COMMAND:L{line_number}] {command}")
            self.logged_lines.add(line_number)
            return True
        return False
        
    def stop_tracing(self):
        """Stop tracing and restore original trace function."""
        if hasattr(self, 'original_trace'):
            sys.settrace(self.original_trace)
            
    def start_tracing(self):
        """Start command tracking using stack inspection approach."""
        # Since sys.settrace doesn't work reliably, we'll use a different approach
        # We'll log commands proactively by checking execution context
        pass
            
    def _backup_trace(self, frame, event, arg):
        """Backup trace function to catch commands that don't produce output."""
        try:
            if (event == 'line' and 
                self.script_filename and 
                frame.f_code.co_filename == self.script_filename):
                
                line_number = frame.f_lineno
                # Debug: log every line we're tracing
                if hasattr(self, 'debug_count'):
                    self.debug_count += 1
                else:
                    self.debug_count = 1
                    
                # Only log first few traces to avoid spam
                if self.debug_count <= 5:
                    self.logger.info(f"[DEBUG] Tracing line {line_number}")
                
                self.log_command_if_meaningful(line_number)
                    
        except Exception as e:
            # Log exceptions for debugging
            self.logger.info(f"[DEBUG] Trace exception: {e}")
            
        # Continue with original trace if it exists
        if hasattr(self, 'original_trace') and self.original_trace:
            return self.original_trace(frame, event, arg)
        return self._backup_trace
        
    def _is_meaningful_line(self, source_line):
        """Check if a source line contains meaningful commands to log."""
        stripped = source_line.strip()
        
        # Skip empty lines and comments
        if not stripped or stripped.startswith('#'):
            return False
            
        # Skip setup and import lines (be more specific to avoid false negatives)
        skip_patterns = [
            'import sys',
            'from pathlib import Path',
            'import pandas',
            'import numpy',
            'from utils.logging_utils import start_logging',
            'sys.path.append',
            'start_logging(',
            'logger =',
            '# ',
            '"""',
            "'''",
            '########',  # Comment separators
        ]
        
        # Check for exact matches or patterns that should be skipped
        for pattern in skip_patterns:
            if pattern in stripped:
                return False
        
        # Skip simple variable assignments for paths (but allow other assignments)
        path_assignments = ['PERSONAL =', 'CODE_DIR =', 'DATA =', 'OUTPUT =']
        for pattern in path_assignments:
            if stripped.startswith(pattern):
                return False
                
        # Include ALL meaningful Python statements (much more comprehensive)
        # Any line that contains executable code should be logged
        
        # Skip only truly non-executable lines
        non_executable_patterns = [
            stripped.startswith('"""'),  # Docstrings
            stripped.startswith("'''"),  # Docstrings
            stripped == 'pass',  # Pass statements
            stripped.startswith('@'),  # Decorators (usually)
            stripped.endswith(':') and len(stripped.split()) == 1,  # Bare colons
        ]
        
        if any(non_executable_patterns):
            return False
            
        # If it's not empty, not a comment, not an import/setup, then it's meaningful
        # This captures: function calls, assignments, loops, conditionals, expressions, etc.
        return True


class StreamCapture:
    """
    Enhanced stream capture that preserves original functionality while logging output.
    Provides intelligent command detection and improved content filtering.
    """
    def __init__(self, logger, original_stream, stream_name="OUTPUT", log_level=logging.INFO, command_tracker=None):
        self.logger = logger
        self.original_stream = original_stream
        self.stream_name = stream_name
        self.log_level = log_level
        self.buffer = []  # Buffer for multi-line content
        self.command_patterns = self._initialize_command_patterns()
        self.command_tracker = command_tracker
        
    def _initialize_command_patterns(self):
        """Initialize patterns that indicate command execution."""
        return {
            'function_calls': [
                r'^\s*\w+\(',  # Function calls starting lines
                r'^\s*\w+\.\w+\(',  # Method calls
                r'^\s*print\(',  # Print statements
            ],
            'assignments': [
                r'^\s*\w+\s*=',  # Variable assignments
                r'^\s*\w+\[.*\]\s*=',  # Index assignments
            ],
            'imports': [
                r'^\s*import\s+',  # Import statements
                r'^\s*from\s+.*\s+import',  # From imports
            ],
            'control_flow': [
                r'^\s*(if|elif|else|for|while|try|except|finally|with)\s',
                r'^\s*(def|class)\s+\w+',  # Function/class definitions
            ]
        }
        
    def write(self, content):
        """Write content to both original stream and logger with enhanced filtering."""
        # Always write to original stream to preserve functionality
        if hasattr(self.original_stream, 'write'):
            self.original_stream.write(content)
            
        # Enhanced content processing
        if content:
            self._process_content(content)
    
    def _process_content(self, content):
        """Process content with enhanced command detection and filtering."""
        # Handle multi-line content by buffering
        if '\n' in content:
            # Split content and process each line
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if i == len(lines) - 1 and not line:
                    # Skip empty last line from split
                    continue
                self._process_line(line)
        else:
            self._process_line(content)
    
    def _process_line(self, line):
        """Process individual line with command detection."""
        if not line.strip():
            return
            
        # Filter out noise first
        if self._is_noise(line):
            return
        
        # CRITICAL FIX: Log pending commands BEFORE processing any output
        # This ensures command logging happens before output logging
        if self.command_tracker:
            self._log_pending_commands()
            
        # Detect command patterns
        command_type = self._detect_command_pattern(line)
        
        if command_type:
            # This looks like a command
            self.logger.log(self.log_level, f"[COMMAND] {line.strip()}")
        else:
            # This looks like output - commands should already be logged by now
            self.logger.log(self.log_level, f"[{self.stream_name}] {line.strip()}")
            
    def _log_pending_commands(self):
        """Aggressively check for and log any pending commands."""
        if not self.command_tracker:
            return
            
        # Get the current execution context
        try:
            frame = inspect.currentframe()
            if frame:
                # Walk up the stack to find the main script frame
                # Go deeper into the stack to find the actual executing line
                current_frame = frame.f_back.f_back.f_back  # Go up past this method, _process_line, and _process_content
                while current_frame:
                    filename = current_frame.f_code.co_filename
                    if filename == self.command_tracker.script_filename:
                        line_number = current_frame.f_lineno
                        
                        # Log this command and any commands leading up to it
                        # Also check the previous line in case we're in the middle of execution
                        self._log_commands_up_to_line(line_number + 1)  # +1 to catch the current executing line
                        break
                    current_frame = current_frame.f_back
        except Exception:
            pass
            
    def _log_commands_up_to_line(self, current_line):
        """Log all meaningful commands up to the current line that haven't been logged yet."""
        if not self.command_tracker:
            return
            
        # Find all unlogged commands that should have executed by now
        for line_num in sorted(self.command_tracker.meaningful_commands.keys()):
            if (line_num <= current_line and 
                line_num not in self.command_tracker.logged_lines):
                
                # Check if this line is part of a code block
                if (self.command_tracker.block_analyzer and 
                    hasattr(self.command_tracker, 'logged_blocks')):
                    
                    block = self.command_tracker.block_analyzer.get_block_for_line(line_num)
                    if block and block['start_line'] not in self.command_tracker.logged_blocks:
                        # Log the entire code block
                        self._log_code_block(block)
                        self.command_tracker.logged_blocks.add(block['start_line'])
                        # Mark all lines in the block as logged
                        for block_line in block['lines']:
                            self.command_tracker.logged_lines.add(block_line)
                    elif not block:
                        # Log individual command (not part of a block)
                        command = self.command_tracker.meaningful_commands[line_num]
                        self.logger.info(f"[COMMAND:L{line_num}] {command}")
                        self.command_tracker.logged_lines.add(line_num)
                else:
                    # Fallback to individual command logging
                    command = self.command_tracker.meaningful_commands[line_num]
                    self.logger.info(f"[COMMAND:L{line_num}] {command}")
                    self.command_tracker.logged_lines.add(line_num)
                    
    def _log_code_block(self, block):
        """Log an entire code block as a grouped unit."""
        block_type = block['type'].upper()
        start_line = block['start_line']
        end_line = block['end_line']
        code = block['code']
        
        # Log the code block with proper formatting
        self.logger.info(f"[{block_type} BLOCK:L{start_line}-{end_line}]")
        for line in code.split('\n'):
            if line.strip():  # Only log non-empty lines
                self.logger.info(f"  {line}")
        self.logger.info(f"[/{block_type} BLOCK]")
    
    def _detect_command_pattern(self, line):
        """Detect if a line contains a command pattern."""
        import re
        
        stripped_line = line.strip()
        if not stripped_line:
            return None
            
        # Check each category of command patterns
        for category, patterns in self.command_patterns.items():
            for pattern in patterns:
                if re.match(pattern, stripped_line):
                    return category
                    
        # Additional heuristics for command detection
        if self._looks_like_command(stripped_line):
            return 'detected_command'
            
        return None
    
    def _looks_like_command(self, line):
        """Additional heuristics to detect commands."""
        # Check for common command indicators
        command_indicators = [
            line.endswith(':'),  # Control flow statements
            '=' in line and not line.startswith(' '),  # Assignments at start of line
            line.startswith('>>>') or line.startswith('...'),  # Interactive prompts
            any(keyword in line for keyword in ['def ', 'class ', 'import ', 'from ']),
        ]
        
        return any(command_indicators)
    
    def flush(self):
        """Flush both original stream and any buffered content."""
        if hasattr(self.original_stream, 'flush'):
            self.original_stream.flush()
            
        # Process any remaining buffered content
        if self.buffer:
            for buffered_content in self.buffer:
                self._process_content(buffered_content)
            self.buffer.clear()
    
    def _is_noise(self, content):
        """Enhanced noise filtering to distinguish between commands and outputs."""
        noise_patterns = [
            '\n\n\n',  # Multiple empty lines
            '...',      # Progress indicators (but not Python continuation)
            '\r',       # Carriage returns without content
            '\x1b[',    # ANSI escape sequences
        ]
        
        # Check if content is just whitespace
        stripped = content.strip()
        if not stripped:
            return True
            
        # Check against noise patterns
        for pattern in noise_patterns:
            if pattern in content:
                return True
                
        # Filter out repetitive output patterns
        if len(stripped) == 1 and stripped in '.-+*':
            return True
            
        # Filter out progress bars and similar
        if all(c in '=->|\\/' for c in stripped.replace(' ', '')):
            return True
                
        return False
    
    def restore(self):
        """Restore the original stream."""
        return self.original_stream
    
    # Delegate other stream methods to original stream
    def __getattr__(self, name):
        return getattr(self.original_stream, name)


class EnhancedLogger:
    """
    Enhanced logger that provides optional manual logging methods.
    """
    def __init__(self, base_logger):
        self.logger = base_logger
    
    # Delegate logging methods to the base logger
    def info(self, msg):
        """Log info message."""
        self.logger.info(msg)
        
    def debug(self, msg):
        """Log debug message."""
        self.logger.debug(msg)
        
    def warning(self, msg):
        """Log warning message."""
        self.logger.warning(msg)
        
    def error(self, msg):
        """Log error message."""
        self.logger.error(msg)
        
    def critical(self, msg):
        """Log critical message."""
        self.logger.critical(msg)


def setup_logging(log_file: str, log_dir: str = 'output/logs') -> EnhancedLogger:
    """
    Set up simple automatic logging with minimal configuration.
    
    This function creates a logging setup that automatically captures:
    - All print statements and console output
    - Command execution with enhanced detection
    - Everything is logged without manual logging calls
    
    Args:
        log_file: Name of the log file
        log_dir: Directory to store log files (will be created if it doesn't exist)
        
    Returns:
        Enhanced logger instance for optional manual logging
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a unique logger name to avoid conflicts
    logger_name = f"script_logger_{log_file.replace('.', '_')}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create file handler with detailed format
    file_handler = logging.FileHandler(
        filename=os.path.join(log_dir, log_file),
        mode='w'
    )
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Add only file handler to the main logger (no console handler)
    logger.addHandler(file_handler)
    
    # Try to get the calling script filename for command tracking
    script_filename = None
    frame = inspect.currentframe()
    try:
        # Walk up the call stack to find the main script
        current_frame = frame
        while current_frame:
            filename = current_frame.f_code.co_filename
            # Look for a file that's not this logging module
            if filename != __file__ and not filename.endswith('logging_utils.py'):
                script_filename = filename
                break
            current_frame = current_frame.f_back
        
        if script_filename:
            # Create and start command tracker
            command_tracker = CommandTracker(logger)
            command_tracker.set_script_filename(script_filename)
            command_tracker.start_tracing()
        else:
            command_tracker = None
    except Exception:
        command_tracker = None
    finally:
        del frame
    
    # Automatically redirect stdout and stderr to capture all output
    # This preserves console output while logging everything to file
    sys.stdout = StreamCapture(logger, sys.stdout, "STDOUT", logging.INFO, command_tracker)
    sys.stderr = StreamCapture(logger, sys.stderr, "STDERR", logging.ERROR, command_tracker)
    
    # Create and return enhanced logger
    enhanced_logger = EnhancedLogger(logger)
    
    # Log the setup
    enhanced_logger.info("=" * 50)
    enhanced_logger.info(f"AUTOMATIC LOGGING INITIALIZED - {log_file}")
    enhanced_logger.info("=" * 50)
    
    return enhanced_logger


def start_logging(log_file: str = "", log_dir: str = "") -> EnhancedLogger:
    """
    Main entry point for automatic logging system.
    
    This function provides a simple interface to enable comprehensive logging
    of script execution flow and results with sensible defaults.
    
    Args:
        log_file: Name of the log file. If empty, defaults to script_name.log
        log_dir: Directory to store log files. If empty, defaults to output/logs
        
    Returns:
        Enhanced logger instance compatible with existing code
        
    Requirements addressed:
        - 1.1: Automatically capture all subsequent command executions and outputs
        - 1.4: Use sensible defaults for log file location when no parameters provided
        - 1.5: Return logger instance compatible with existing code
    """
    # Handle default log_file parameter
    if not log_file:
        # Get the calling script's filename to create a default log file name
        frame = inspect.currentframe()
        try:
            # Go up the call stack to find the calling script
            caller_frame = frame.f_back
            if caller_frame and caller_frame.f_code.co_filename != __file__:
                script_path = Path(caller_frame.f_code.co_filename)
                log_file = f"{script_path.stem}.log"
            else:
                # Fallback if we can't determine the script name
                log_file = "script_execution.log"
        finally:
            del frame  # Prevent reference cycles
    
    # Handle default log_dir parameter
    if not log_dir:
        log_dir = "output/logs"
    
    # Convert Path objects to strings if needed
    if isinstance(log_dir, Path):
        log_dir = str(log_dir)
    
    # Use the existing setup_logging function to maintain compatibility
    return setup_logging(log_file=log_file, log_dir=log_dir)