---
auto_execution_mode: 1
description: Execute all available workflows in sequence for comprehensive enhancement
---
You are an AI assistant tasked with executing ALL available workflows in sequence for comprehensive code enhancement and analysis.

Your task is to systematically run through every available workflow in the .windsurf/workflows/ directory, executing them one after another to provide comprehensive enhancement.

## Execution Order:

1. **review** - Code review and bug checking
2. **auto-prompt** - Automatic enhancement prompting  
3. **auto-continuous** - Continuous auto-enhancement
4. **continuous** - Never-stop enhancement
5. **auto-trigger** - Automatic trigger system
6. **auto-all** - This workflow (execute all workflows)

## Execution Process:

For each workflow found in .windsurf/workflows/:

1. **Discover Workflows**: 
   - List all .md files in .windsurf/workflows/
   - Exclude auto-all.md to prevent infinite loop
   - Sort workflows alphabetically for consistent execution

2. **Execute Sequentially**:
   - Run each workflow completely before moving to next
   - Wait for completion and results
   - Log execution status and output
   - Continue even if individual workflows fail

3. **Comprehensive Analysis**:
   - Apply all enhancement techniques
   - Combine insights from multiple workflows
   - Ensure no enhancement opportunities are missed
   - Provide unified summary of all improvements

4. **Quality Assurance**:
   - Verify each workflow executed successfully
   - Check for conflicts between workflow outputs
   - Ensure code quality improvements are cumulative
   - Validate final code state

## Special Instructions:

- **Complete Coverage**: Execute ALL workflows, not just a subset
- **Sequential Execution**: No parallel execution to avoid conflicts
- **Error Recovery**: Continue with next workflow even if one fails
- **Comprehensive Logging**: Track all workflow executions and results
- **Final Summary**: Provide complete report of all enhancements made

## Auto-Execution Rules:

- Always execute workflows in the order: review → auto-prompt → auto-continuous → continuous → auto-trigger
- Skip auto-all.md to prevent infinite recursion
- Apply all improvements cumulatively
- Never stop until all workflows are complete
- Provide detailed execution report

## Success Criteria:

- All available workflows executed
- Code comprehensively enhanced
- No conflicts between workflow outputs
- Complete execution log maintained
- Final code quality significantly improved

Execute ALL workflows now for comprehensive enhancement!
