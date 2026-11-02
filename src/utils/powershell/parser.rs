//! PowerShell command parser (adapted from src/utils/powershell/parser.ts).
//! Equivalent to codemason/utils/powershell/parser.py.
//!
//! Key architecture (faithful to the source): we do NOT reimplement a
//! PowerShell parser. PARSE_SCRIPT_BODY is a PowerShell script that runs the
//! platform's own `[System.Management.Automation.Language.Parser]::ParseInput`
//! and emits a compact JSON AST; this module spawns PowerShell to run it, then
//! transforms that JSON into typed structures and provides analysis accessors
//! the security/permission layers consume.
//!
//! The command is passed as a UTF-8 Base64 `$EncodedCommand` variable (no
//! here-string injection), and the whole parse script is handed to PowerShell
//! via `-EncodedCommand` (UTF-16LE Base64) — no temp files, no escaping issues.

use base64::Engine;
use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::Value;
use std::collections::{HashMap, HashSet, VecDeque};
use std::io::Read;

use crate::utils::powershell_detect::get_cached_powershell_path;

// =============================================================================
// The PowerShell parse script. Inlined (not a data file) so it always ships
// with the binary. Body is byte-identical to the source's PARSE_SCRIPT_BODY.
// =============================================================================
pub const PARSE_SCRIPT_BODY: &str = r##"
if (-not $EncodedCommand) {
    Write-Output '{"valid":false,"errors":[{"message":"No command provided","errorId":"NoInput"}],"statements":[],"variables":[],"hasStopParsing":false,"originalCommand":""}'
    exit 0
}

$Command = [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String($EncodedCommand))

$tokens = $null
$parseErrors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseInput(
    $Command,
    [ref]$tokens,
    [ref]$parseErrors
)

$allVariables = [System.Collections.ArrayList]::new()

function Get-RawCommandElements {
    param([System.Management.Automation.Language.CommandAst]$CmdAst)
    $elems = [System.Collections.ArrayList]::new()
    foreach ($ce in $CmdAst.CommandElements) {
        $ceData = @{ type = $ce.GetType().Name; text = $ce.Extent.Text }
        if ($ce.PSObject.Properties['Value'] -and $null -ne $ce.Value -and $ce.Value -is [string]) {
            $ceData.value = $ce.Value
        }
        if ($ce -is [System.Management.Automation.Language.CommandExpressionAst]) {
            $ceData.expressionType = $ce.Expression.GetType().Name
        }
        $a=$ce.Argument;if($a){$ceData.children=@(@{type=$a.GetType().Name;text=$a.Extent.Text})}
        [void]$elems.Add($ceData)
    }
    return $elems
}

function Get-RawRedirections {
    param($Redirections)
    $result = [System.Collections.ArrayList]::new()
    foreach ($redir in $Redirections) {
        $redirData = @{ type = $redir.GetType().Name }
        if ($redir -is [System.Management.Automation.Language.FileRedirectionAst]) {
            $redirData.append = [bool]$redir.Append
            $redirData.fromStream = $redir.FromStream.ToString()
            $redirData.locationText = $redir.Location.Extent.Text
        }
        [void]$result.Add($redirData)
    }
    return $result
}

function Get-SecurityPatterns($A) {
    $p = @{}
    foreach ($n in $A.FindAll({ param($x)
        $x -is [System.Management.Automation.Language.MemberExpressionAst] -or
        $x -is [System.Management.Automation.Language.SubExpressionAst] -or
        $x -is [System.Management.Automation.Language.ArrayExpressionAst] -or
        $x -is [System.Management.Automation.Language.ExpandableStringExpressionAst] -or
        $x -is [System.Management.Automation.Language.ScriptBlockExpressionAst] -or
        $x -is [System.Management.Automation.Language.ParenExpressionAst]
    }, $true)) { switch ($n.GetType().Name) {
        'InvokeMemberExpressionAst' { $p.hasMemberInvocations = $true }
        'MemberExpressionAst' { $p.hasMemberInvocations = $true }
        'SubExpressionAst' { $p.hasSubExpressions = $true }
        'ArrayExpressionAst' { $p.hasSubExpressions = $true }
        'ParenExpressionAst' { $p.hasSubExpressions = $true }
        'ExpandableStringExpressionAst' { $p.hasExpandableStrings = $true }
        'ScriptBlockExpressionAst' { $p.hasScriptBlocks = $true }
    }}
    if ($p.Count -gt 0) { return $p }
    return $null
}

$varExprs = $ast.FindAll({ param($node) $node -is [System.Management.Automation.Language.VariableExpressionAst] }, $true)
foreach ($v in $varExprs) {
    [void]$allVariables.Add(@{
        path = $v.VariablePath.ToString()
        isSplatted = [bool]$v.Splatted
    })
}

$typeLiterals = [System.Collections.ArrayList]::new()
foreach ($t in $ast.FindAll({ param($n)
    $n -is [System.Management.Automation.Language.TypeExpressionAst] -or
    $n -is [System.Management.Automation.Language.TypeConstraintAst]
}, $true)) { [void]$typeLiterals.Add($t.TypeName.FullName) }

$hasStopParsing = $false
$tk = [System.Management.Automation.Language.TokenKind]
foreach ($tok in $tokens) {
    if ($tok.Kind -eq $tk::MinusMinus) { $hasStopParsing = $true; break }
    if ($tok.Kind -eq $tk::Generic -and ($tok.Text -replace '[\u2013\u2014\u2015]','-') -eq '--%') {
        $hasStopParsing = $true; break
    }
}

$statements = [System.Collections.ArrayList]::new()

function Process-BlockStatements {
    param($Block)
    if (-not $Block) { return }

    foreach ($stmt in $Block.Statements) {
        $statement = @{
            type = $stmt.GetType().Name
            text = $stmt.Extent.Text
        }

        if ($stmt -is [System.Management.Automation.Language.PipelineAst]) {
            $elements = [System.Collections.ArrayList]::new()
            foreach ($element in $stmt.PipelineElements) {
                $elemData = @{
                    type = $element.GetType().Name
                    text = $element.Extent.Text
                }

                if ($element -is [System.Management.Automation.Language.CommandAst]) {
                    $elemData.commandElements = @(Get-RawCommandElements -CmdAst $element)
                    $elemData.redirections = @(Get-RawRedirections -Redirections $element.Redirections)
                } elseif ($element -is [System.Management.Automation.Language.CommandExpressionAst]) {
                    $elemData.expressionType = $element.Expression.GetType().Name
                    $elemData.redirections = @(Get-RawRedirections -Redirections $element.Redirections)
                }

                [void]$elements.Add($elemData)
            }
            $statement.elements = @($elements)

            $allNestedCmds = $stmt.FindAll(
                { param($node) $node -is [System.Management.Automation.Language.CommandAst] },
                $true
            )
            $nestedCmds = [System.Collections.ArrayList]::new()
            foreach ($cmd in $allNestedCmds) {
                if ($cmd.Parent -eq $stmt) { continue }
                $nested = @{
                    type = $cmd.GetType().Name
                    text = $cmd.Extent.Text
                    commandElements = @(Get-RawCommandElements -CmdAst $cmd)
                    redirections = @(Get-RawRedirections -Redirections $cmd.Redirections)
                }
                [void]$nestedCmds.Add($nested)
            }
            if ($nestedCmds.Count -gt 0) {
                $statement.nestedCommands = @($nestedCmds)
            }
            $r = $stmt.FindAll({param($n) $n -is [System.Management.Automation.Language.FileRedirectionAst]}, $true)
            if ($r.Count -gt 0) {
                $rr = @(Get-RawRedirections -Redirections $r)
                $statement.redirections = if ($statement.redirections) { @($statement.redirections) + $rr } else { $rr }
            }
        } else {
            $nestedCmdAsts = $stmt.FindAll(
                { param($node) $node -is [System.Management.Automation.Language.CommandAst] },
                $true
            )
            $nested = [System.Collections.ArrayList]::new()
            foreach ($cmd in $nestedCmdAsts) {
                [void]$nested.Add(@{
                    type = 'CommandAst'
                    text = $cmd.Extent.Text
                    commandElements = @(Get-RawCommandElements -CmdAst $cmd)
                    redirections = @(Get-RawRedirections -Redirections $cmd.Redirections)
                })
            }
            if ($nested.Count -gt 0) {
                $statement.nestedCommands = @($nested)
            }
            $r = $stmt.FindAll({param($n) $n -is [System.Management.Automation.Language.FileRedirectionAst]}, $true)
            if ($r.Count -gt 0) { $statement.redirections = @(Get-RawRedirections -Redirections $r) }
        }

        $sp = Get-SecurityPatterns $stmt
        if ($sp) { $statement.securityPatterns = $sp }

        [void]$statements.Add($statement)
    }

    if ($Block.Traps) {
        foreach ($trap in $Block.Traps) {
            $statement = @{
                type = 'TrapStatementAst'
                text = $trap.Extent.Text
            }
            $nestedCmdAsts = $trap.FindAll(
                { param($node) $node -is [System.Management.Automation.Language.CommandAst] },
                $true
            )
            $nestedCmds = [System.Collections.ArrayList]::new()
            foreach ($cmd in $nestedCmdAsts) {
                $nested = @{
                    type = $cmd.GetType().Name
                    text = $cmd.Extent.Text
                    commandElements = @(Get-RawCommandElements -CmdAst $cmd)
                    redirections = @(Get-RawRedirections -Redirections $cmd.Redirections)
                }
                [void]$nestedCmds.Add($nested)
            }
            if ($nestedCmds.Count -gt 0) {
                $statement.nestedCommands = @($nestedCmds)
            }
            $r = $trap.FindAll({param($n) $n -is [System.Management.Automation.Language.FileRedirectionAst]}, $true)
            if ($r.Count -gt 0) { $statement.redirections = @(Get-RawRedirections -Redirections $r) }
            $sp = Get-SecurityPatterns $trap
            if ($sp) { $statement.securityPatterns = $sp }
            [void]$statements.Add($statement)
        }
    }
}

Process-BlockStatements -Block $ast.BeginBlock
Process-BlockStatements -Block $ast.ProcessBlock
Process-BlockStatements -Block $ast.EndBlock
Process-BlockStatements -Block $ast.CleanBlock
Process-BlockStatements -Block $ast.DynamicParamBlock

if ($ast.ParamBlock) {
  $pb = $ast.ParamBlock
  $pn = [System.Collections.ArrayList]::new()
  foreach ($c in $pb.FindAll({param($n) $n -is [System.Management.Automation.Language.CommandAst]}, $true)) {
    [void]$pn.Add(@{type='CommandAst';text=$c.Extent.Text;commandElements=@(Get-RawCommandElements -CmdAst $c);redirections=@(Get-RawRedirections -Redirections $c.Redirections)})
  }
  $pr = $pb.FindAll({param($n) $n -is [System.Management.Automation.Language.FileRedirectionAst]}, $true)
  $ps = Get-SecurityPatterns $pb
  if ($pn.Count -gt 0 -or $pr.Count -gt 0 -or $ps) {
    $st = @{type='ParamBlockAst';text=$pb.Extent.Text}
    if ($pn.Count -gt 0) { $st.nestedCommands = @($pn) }
    if ($pr.Count -gt 0) { $st.redirections = @(Get-RawRedirections -Redirections $pr) }
    if ($ps) { $st.securityPatterns = $ps }
    [void]$statements.Add($st)
  }
}

$hasUsingStatements = $ast.UsingStatements -and $ast.UsingStatements.Count -gt 0
$hasScriptRequirements = $ast.ScriptRequirements -ne $null

$output = @{
    valid = ($parseErrors.Count -eq 0)
    errors = @($parseErrors | ForEach-Object {
        @{
            message = $_.Message
            errorId = $_.ErrorId
        }
    })
    statements = @($statements)
    variables = @($allVariables)
    hasStopParsing = $hasStopParsing
    originalCommand = $Command
    typeLiterals = @($typeLiterals)
    hasUsingStatements = [bool]$hasUsingStatements
    hasScriptRequirements = [bool]$hasScriptRequirements
}

$output | ConvertTo-Json -Depth 10 -Compress
"##;

// =============================================================================
// Command-length budget (derived from PARSE_SCRIPT_BODY length; UTF-8 BYTES).
// Windows CreateProcess caps the command line at 32,767 chars. The encoding
// chain (command → UTF-8 b64 → wrapper → script → UTF-16LE → b64 → argv)
// inflates ~8/3×, so the safe UTF-8 byte budget is computed from the script
// length and MUST be compared against the command's UTF-8 byte length, never
// its code-point count — a CJK char is 1 code point but 3 UTF-8 bytes
// (source Finding #36).
// =============================================================================
const WINDOWS_ARGV_CAP: f64 = 32_767.0;
const FIXED_ARGV_OVERHEAD: f64 = 200.0;
const ENCODED_CMD_WRAPPER: f64 = 21.0; // len("$EncodedCommand = ''\n")
const SAFETY_MARGIN: i64 = 100;

pub static WINDOWS_MAX_COMMAND_LENGTH: Lazy<usize> = Lazy::new(|| {
    let script_chars_budget = (WINDOWS_ARGV_CAP - FIXED_ARGV_OVERHEAD) * 3.0 / 8.0;
    let cmd_b64_budget =
        script_chars_budget - PARSE_SCRIPT_BODY.len() as f64 - ENCODED_CMD_WRAPPER;
    let raw = ((cmd_b64_budget * 3.0) / 4.0) as i64 - SAFETY_MARGIN;
    raw.max(0) as usize
});

/// Pre-existing Unix value (argv limits are ~128KB/arg; far above this). The
/// Windows derivation must NOT be applied on Unix (would regress mid-size
/// commands that parse fine and reach the deny loop). Platform-gated. UTF-8 BYTES.
const UNIX_MAX_COMMAND_LENGTH: usize = 4_500;

pub fn max_command_length() -> usize {
    if cfg!(windows) {
        *WINDOWS_MAX_COMMAND_LENGTH
    } else {
        UNIX_MAX_COMMAND_LENGTH
    }
}

/// Warm-spawn is ~450ms; 5s covers loaded hosts. Override via env for tests.
const DEFAULT_PARSE_TIMEOUT_MS: u64 = 5_000;

fn get_parse_timeout_ms() -> u64 {
    if let Ok(env) = std::env::var("CODEMASON_PWSH_PARSE_TIMEOUT_MS") {
        if let Ok(v) = env.parse::<i64>() {
            if v > 0 {
                return v as u64;
            }
        }
    }
    DEFAULT_PARSE_TIMEOUT_MS
}

// =============================================================================
// Typed output structures (ParsedPowerShellCommand and friends).
// =============================================================================
#[derive(Debug, Clone, PartialEq)]
pub struct CommandElementChild {
    pub element_type: String, // CommandElementType ('type' in Python)
    pub text: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParsedRedirection {
    pub operator: String, // '>' | '>>' | '2>' | '2>>' | '*>' | '*>>' | '2>&1'
    pub target: String,
    pub is_merging: bool,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ParsedCommandElement {
    pub name: String,
    pub name_type: String, // 'cmdlet' | 'application' | 'unknown'
    pub element_type: String, // PipelineElementType
    pub args: Vec<String>,
    pub text: String,
    pub element_types: Option<Vec<String>>, // CommandElementType[]
    pub children: Option<Vec<Option<Vec<CommandElementChild>>>>,
    pub redirections: Option<Vec<ParsedRedirection>>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ParsedStatement {
    pub statement_type: String, // StatementType
    pub commands: Vec<ParsedCommandElement>,
    pub redirections: Vec<ParsedRedirection>,
    pub text: String,
    pub nested_commands: Option<Vec<ParsedCommandElement>>,
    pub security_patterns: Option<HashMap<String, bool>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParsedVariable {
    pub path: String,
    pub is_splatted: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParseError {
    pub message: String,
    pub error_id: String,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ParsedPowerShellCommand {
    pub valid: bool,
    pub errors: Vec<ParseError>,
    pub statements: Vec<ParsedStatement>,
    pub variables: Vec<ParsedVariable>,
    pub has_stop_parsing: bool,
    pub original_command: String,
    pub type_literals: Option<Vec<String>>,
    pub has_using_statements: Option<bool>,
    pub has_script_requirements: Option<bool>,
}

fn make_invalid_result(command: &str, message: &str, error_id: &str) -> ParsedPowerShellCommand {
    ParsedPowerShellCommand {
        valid: false,
        errors: vec![ParseError { message: message.to_string(), error_id: error_id.to_string() }],
        statements: vec![],
        variables: vec![],
        has_stop_parsing: false,
        original_command: command.to_string(),
        ..Default::default()
    }
}

/// PowerShell 5.1's ConvertTo-Json unwraps single-element arrays into plain
/// objects; normalise to a list (ensureArray).
fn ensure_array(value: Option<&Value>) -> Vec<Value> {
    match value {
        None | Some(Value::Null) => vec![],
        Some(Value::Array(a)) => a.clone(),
        Some(other) => vec![other.clone()],
    }
}

// =============================================================================
// Encoding + script assembly.
// =============================================================================

/// Base64 of UTF-16LE (required by PowerShell -EncodedCommand).
pub(crate) fn to_utf16le_base64(text: &str) -> String {
    let mut bytes = Vec::with_capacity(text.len() * 2);
    for unit in text.encode_utf16() {
        bytes.extend_from_slice(&unit.to_le_bytes());
    }
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// Prefix the UTF-8 Base64 $EncodedCommand var onto PARSE_SCRIPT_BODY.
pub fn build_parse_script(command: &str) -> String {
    let encoded = base64::engine::general_purpose::STANDARD.encode(command.as_bytes());
    format!("$EncodedCommand = '{encoded}'\n{PARSE_SCRIPT_BODY}")
}

// =============================================================================
// Raw .NET AST type name → our union mappers.
// =============================================================================
const STATEMENT_TYPES: &[&str] = &[
    "PipelineAst", "PipelineChainAst", "AssignmentStatementAst", "IfStatementAst",
    "ForStatementAst", "ForEachStatementAst", "WhileStatementAst", "DoWhileStatementAst",
    "DoUntilStatementAst", "SwitchStatementAst", "TryStatementAst", "TrapStatementAst",
    "FunctionDefinitionAst", "DataStatementAst",
];

pub fn map_statement_type(raw_type: &str) -> String {
    if STATEMENT_TYPES.contains(&raw_type) {
        raw_type.to_string()
    } else {
        "UnknownStatementAst".to_string()
    }
}

/// Map raw AST node type to CommandElementType.
pub fn map_element_type(raw_type: &str, expression_type: Option<&str>) -> String {
    match raw_type {
        "ScriptBlockExpressionAst" => "ScriptBlock".to_string(),
        // ArrayExpressionAst (@()) is a sibling of SubExpressionAst; both run
        // arbitrary pipelines with side effects. Map both → SubExpression.
        "SubExpressionAst" | "ArrayExpressionAst" => "SubExpression".to_string(),
        "ExpandableStringExpressionAst" => "ExpandableString".to_string(),
        "InvokeMemberExpressionAst" | "MemberExpressionAst" => "MemberInvocation".to_string(),
        "VariableExpressionAst" => "Variable".to_string(),
        // ConstantExpressionAst covers numeric literals — inert values, treat
        // as StringConstant so numeric args don't false-positive as 'Other'.
        "StringConstantExpressionAst" | "ConstantExpressionAst" => "StringConstant".to_string(),
        "CommandParameterAst" => "Parameter".to_string(),
        "ParenExpressionAst" => "SubExpression".to_string(),
        // Delegate to the wrapped expression type.
        "CommandExpressionAst" => match expression_type {
            Some(et) => map_element_type(et, None),
            None => "Other".to_string(),
        },
        _ => "Other".to_string(),
    }
}

static CMDLET_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^[A-Za-z]+-[A-Za-z][A-Za-z0-9_]*$").unwrap());
static PATHISH_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"[.\\/]").unwrap());
static DRIVE_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^[A-Za-z]:").unwrap());
/// Any non-ASCII char, all planes — so the .NET OrdinalIgnoreCase cmdlet-spoof
/// gate cannot be slipped (e.g. a math-bold cmdlet spoof above U+FFFF).
static NONASCII_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"[^\x00-\x7f]").unwrap());
static QUOTE_STRIP_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r#"^['"]|['"]\z"#).unwrap());

/// Classify a command name as cmdlet | application | unknown.
pub fn classify_command_name(name: &str) -> &'static str {
    if CMDLET_RE.is_match(name) {
        "cmdlet"
    } else if PATHISH_RE.is_match(name) {
        "application"
    } else {
        "unknown"
    }
}

/// Strip a module prefix (Module\Cmdlet → Cmdlet), but never a real path.
pub fn strip_module_prefix(name: &str) -> String {
    let Some(idx) = name.rfind('\\') else {
        return name.to_string();
    };
    if DRIVE_RE.is_match(name)
        || name.starts_with("\\\\")
        || name.starts_with(".\\")
        || name.starts_with("..\\")
    {
        return name.to_string();
    }
    name[idx + 1..].to_string()
}

// =============================================================================
// Transforms: raw JSON → typed structures.
// =============================================================================
fn v_str(raw: &Value, key: &str) -> String {
    raw.get(key).and_then(|v| v.as_str()).unwrap_or("").to_string()
}

pub fn transform_redirection(raw: &Value) -> ParsedRedirection {
    if raw.get("type").and_then(|v| v.as_str()) == Some("MergingRedirectionAst") {
        return ParsedRedirection { operator: "2>&1".into(), target: String::new(), is_merging: true };
    }
    let append = raw.get("append").and_then(|v| v.as_bool()).unwrap_or(false);
    let from_stream = raw
        .get("fromStream")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("Output");
    let operator = if append {
        match from_stream {
            "Error" => "2>>",
            "All" => "*>>",
            _ => ">>",
        }
    } else {
        match from_stream {
            "Error" => "2>",
            "All" => "*>",
            _ => ">",
        }
    };
    ParsedRedirection {
        operator: operator.to_string(),
        target: v_str(raw, "locationText"),
        is_merging: false,
    }
}

pub fn transform_command_ast(raw: &Value) -> ParsedCommandElement {
    let cmd_elements = ensure_array(raw.get("commandElements"));
    let mut name = String::new();
    let mut args: Vec<String> = Vec::new();
    let mut element_types: Vec<String> = Vec::new();
    let mut children: Vec<Option<Vec<CommandElementChild>>> = Vec::new();
    let mut has_children = false;
    let mut name_type = "unknown".to_string();

    if !cmd_elements.is_empty() {
        let first = &cmd_elements[0];
        let first_type = first.get("type").and_then(|v| v.as_str()).unwrap_or("");
        let is_first_string_literal =
            matches!(first_type, "StringConstantExpressionAst" | "ExpandableStringExpressionAst");
        let first_value = first.get("value").and_then(|v| v.as_str());
        let raw_name_unstripped = match (is_first_string_literal, first_value) {
            (true, Some(v)) => v.to_string(),
            _ => v_str(first, "text"),
        };
        // Strip surrounding quotes so downstream readers see the bare name.
        let raw_name = QUOTE_STRIP_RE.replace_all(&raw_name_unstripped, "").into_owned();
        // Non-ASCII in cmdlet position is suspicious (.NET OrdinalIgnoreCase
        // folds e.g. U+017F→S, which a plain lowercase misses) → force
        // 'application' to gate auto-allow. Source Finding #31.
        if NONASCII_RE.is_match(&raw_name) {
            name_type = "application".to_string();
        } else {
            name_type = classify_command_name(&raw_name).to_string();
        }
        name = strip_module_prefix(&raw_name);
        element_types.push(map_element_type(
            first_type,
            first.get("expressionType").and_then(|v| v.as_str()),
        ));

        for ce in &cmd_elements[1..] {
            let ce_type = ce.get("type").and_then(|v| v.as_str()).unwrap_or("");
            let is_string_literal =
                matches!(ce_type, "StringConstantExpressionAst" | "ExpandableStringExpressionAst");
            let ce_value = ce.get("value").filter(|v| !v.is_null());
            args.push(match (is_string_literal, ce_value) {
                (true, Some(v)) => v.as_str().map(|s| s.to_string()).unwrap_or_else(|| v.to_string()),
                _ => v_str(ce, "text"),
            });
            element_types.push(map_element_type(
                ce_type,
                ce.get("expressionType").and_then(|v| v.as_str()),
            ));
            let raw_children = ensure_array(ce.get("children"));
            if !raw_children.is_empty() {
                has_children = true;
                children.push(Some(
                    raw_children
                        .iter()
                        .map(|c| CommandElementChild {
                            element_type: map_element_type(
                                c.get("type").and_then(|v| v.as_str()).unwrap_or(""),
                                None,
                            ),
                            text: v_str(c, "text"),
                        })
                        .collect(),
                ));
            } else {
                children.push(None);
            }
        }
    }

    let mut result = ParsedCommandElement {
        name,
        name_type,
        element_type: "CommandAst".to_string(),
        args,
        text: v_str(raw, "text"),
        element_types: Some(element_types),
        children: if has_children { Some(children) } else { None },
        redirections: None,
    };
    let raw_redirs = ensure_array(raw.get("redirections"));
    if !raw_redirs.is_empty() {
        result.redirections = Some(raw_redirs.iter().map(transform_redirection).collect());
    }
    result
}

pub fn transform_expression_element(raw: &Value) -> ParsedCommandElement {
    let raw_type = raw.get("type").and_then(|v| v.as_str()).unwrap_or("");
    let element_type = if raw_type == "ParenExpressionAst" {
        "ParenExpressionAst"
    } else {
        "CommandExpressionAst"
    };
    ParsedCommandElement {
        name: v_str(raw, "text"),
        name_type: "unknown".to_string(),
        element_type: element_type.to_string(),
        args: vec![],
        text: v_str(raw, "text"),
        element_types: Some(vec![map_element_type(
            raw_type,
            raw.get("expressionType").and_then(|v| v.as_str()),
        )]),
        children: None,
        redirections: None,
    }
}

pub fn transform_statement(raw: &Value) -> ParsedStatement {
    let statement_type = map_statement_type(raw.get("type").and_then(|v| v.as_str()).unwrap_or(""));
    let mut commands: Vec<ParsedCommandElement> = Vec::new();
    let mut redirections: Vec<ParsedRedirection> = Vec::new();

    if raw.get("elements").map(|v| !v.is_null()).unwrap_or(false) {
        for elem in ensure_array(raw.get("elements")) {
            if elem.get("type").and_then(|v| v.as_str()) == Some("CommandAst") {
                commands.push(transform_command_ast(&elem));
            } else {
                commands.push(transform_expression_element(&elem));
            }
            for redir in ensure_array(elem.get("redirections")) {
                redirections.push(transform_redirection(&redir));
            }
        }
        // Merge the statement-level deep-FindAll redirections, deduped by
        // (operator, target) — they re-discover direct-element ones.
        let mut seen: HashSet<(String, String)> =
            redirections.iter().map(|r| (r.operator.clone(), r.target.clone())).collect();
        for redir in ensure_array(raw.get("redirections")) {
            let r = transform_redirection(&redir);
            let key = (r.operator.clone(), r.target.clone());
            if !seen.contains(&key) {
                seen.insert(key);
                redirections.push(r);
            }
        }
    } else {
        commands.push(ParsedCommandElement {
            name: v_str(raw, "text"),
            name_type: "unknown".to_string(),
            element_type: "CommandExpressionAst".to_string(),
            args: vec![],
            text: v_str(raw, "text"),
            ..Default::default()
        });
        for redir in ensure_array(raw.get("redirections")) {
            redirections.push(transform_redirection(&redir));
        }
    }

    let raw_nested = ensure_array(raw.get("nestedCommands"));
    let nested_commands = if raw_nested.is_empty() {
        None
    } else {
        Some(raw_nested.iter().map(transform_command_ast).collect())
    };

    let security_patterns = raw.get("securityPatterns").and_then(|sp| sp.as_object()).map(|obj| {
        obj.iter()
            .map(|(k, v)| (k.clone(), v.as_bool().unwrap_or(false)))
            .collect::<HashMap<String, bool>>()
    });

    ParsedStatement {
        statement_type,
        commands,
        redirections,
        text: v_str(raw, "text"),
        nested_commands,
        security_patterns: security_patterns.filter(|m| !m.is_empty()),
    }
}

fn transform_raw_output(raw: &Value) -> ParsedPowerShellCommand {
    let mut result = ParsedPowerShellCommand {
        valid: raw.get("valid").and_then(|v| v.as_bool()).unwrap_or(false),
        errors: ensure_array(raw.get("errors"))
            .iter()
            .map(|e| ParseError { message: v_str(e, "message"), error_id: v_str(e, "errorId") })
            .collect(),
        statements: ensure_array(raw.get("statements")).iter().map(transform_statement).collect(),
        variables: ensure_array(raw.get("variables"))
            .iter()
            .map(|v| ParsedVariable {
                path: v_str(v, "path"),
                is_splatted: v.get("isSplatted").and_then(|b| b.as_bool()).unwrap_or(false),
            })
            .collect(),
        has_stop_parsing: raw.get("hasStopParsing").and_then(|v| v.as_bool()).unwrap_or(false),
        original_command: v_str(raw, "originalCommand"),
        ..Default::default()
    };
    let tl: Vec<String> = ensure_array(raw.get("typeLiterals"))
        .iter()
        .filter_map(|v| v.as_str().map(|s| s.to_string()))
        .collect();
    if !tl.is_empty() {
        result.type_literals = Some(tl);
    }
    if raw.get("hasUsingStatements").and_then(|v| v.as_bool()).unwrap_or(false) {
        result.has_using_statements = Some(true);
    }
    if raw.get("hasScriptRequirements").and_then(|v| v.as_bool()).unwrap_or(false) {
        result.has_script_requirements = Some(true);
    }
    result
}

// =============================================================================
// Parse invocation (spawn PowerShell, one retry on timeout, LRU-memoized).
// =============================================================================
const TRANSIENT_ERROR_IDS: &[&str] =
    &["PwshSpawnError", "PwshError", "PwshTimeout", "EmptyOutput", "InvalidJson"];

const PARSE_CACHE_MAX: usize = 256;

struct ParseCache {
    map: HashMap<String, ParsedPowerShellCommand>,
    order: VecDeque<String>, // front = least recently used
}

static PARSE_CACHE: Lazy<parking_lot::Mutex<ParseCache>> =
    Lazy::new(|| parking_lot::Mutex::new(ParseCache { map: HashMap::new(), order: VecDeque::new() }));

enum RunOutcome {
    Done { stdout: String, stderr: String, code: Option<i32> },
    Timeout,
    SpawnError(String),
}

/// Run a command with a wall-clock timeout, draining stdout/stderr on threads
/// so a chatty child can't deadlock on a full pipe.
fn run_with_timeout(exe: &str, args: &[String], timeout: std::time::Duration) -> RunOutcome {
    let mut child = match std::process::Command::new(exe)
        .args(args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => return RunOutcome::SpawnError(e.to_string()),
    };

    let mut stdout_pipe = child.stdout.take();
    let mut stderr_pipe = child.stderr.take();
    let out_handle = std::thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(ref mut p) = stdout_pipe {
            let _ = p.read_to_end(&mut buf);
        }
        buf
    });
    let err_handle = std::thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(ref mut p) = stderr_pipe {
            let _ = p.read_to_end(&mut buf);
        }
        buf
    });

    let deadline = std::time::Instant::now() + timeout;
    let code = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status.code().unwrap_or(-1)),
            Ok(None) => {
                if std::time::Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    let _ = out_handle.join();
                    let _ = err_handle.join();
                    return RunOutcome::Timeout;
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            Err(e) => return RunOutcome::SpawnError(e.to_string()),
        }
    };

    let stdout_bytes = out_handle.join().unwrap_or_default();
    let stderr_bytes = err_handle.join().unwrap_or_default();
    RunOutcome::Done {
        stdout: String::from_utf8_lossy(&stdout_bytes).into_owned(),
        stderr: String::from_utf8_lossy(&stderr_bytes).into_owned(),
        code,
    }
}

fn parse_powershell_command_impl(command: &str) -> ParsedPowerShellCommand {
    // UTF-8 BYTE budget (not code points) — see max_command_length note.
    let command_bytes = command.len();
    let max_len = max_command_length();
    if command_bytes > max_len {
        return make_invalid_result(
            command,
            &format!(
                "Command too long for parsing ({command_bytes} bytes). Maximum supported length is {max_len} bytes."
            ),
            "CommandTooLong",
        );
    }

    let Some(pwsh_path) = get_cached_powershell_path() else {
        return make_invalid_result(command, "PowerShell is not available", "NoPowerShell");
    };

    let script = build_parse_script(command);
    let encoded_script = to_utf16le_base64(&script);
    let args: Vec<String> = vec![
        "-NoProfile".into(),
        "-NonInteractive".into(),
        "-NoLogo".into(),
        "-EncodedCommand".into(),
        encoded_script,
    ];
    let timeout = std::time::Duration::from_millis(get_parse_timeout_ms());

    let mut outcome = RunOutcome::Timeout;
    for _attempt in 0..2 {
        outcome = run_with_timeout(&pwsh_path, &args, timeout);
        match &outcome {
            RunOutcome::SpawnError(e) => {
                return make_invalid_result(
                    command,
                    &format!("Failed to spawn PowerShell: {e}"),
                    "PwshSpawnError",
                );
            }
            RunOutcome::Timeout => continue,
            RunOutcome::Done { .. } => break,
        }
    }

    let (stdout, stderr, code) = match outcome {
        RunOutcome::Timeout => {
            return make_invalid_result(
                command,
                &format!("pwsh timed out after {}ms (2 attempts)", get_parse_timeout_ms()),
                "PwshTimeout",
            );
        }
        RunOutcome::Done { stdout, stderr, code } => (stdout, stderr, code),
        RunOutcome::SpawnError(_) => unreachable!(),
    };

    if code != Some(0) {
        return make_invalid_result(
            command,
            &format!("pwsh exited with code {}: {stderr}", code.map(|c| c.to_string()).unwrap_or_else(|| "None".into())),
            "PwshError",
        );
    }

    let trimmed = stdout.trim();
    if trimmed.is_empty() {
        return make_invalid_result(command, "No output from PowerShell parser", "EmptyOutput");
    }

    match serde_json::from_str::<Value>(trimmed) {
        Ok(raw) => transform_raw_output(&raw),
        Err(_) => make_invalid_result(command, "Invalid JSON from PowerShell parser", "InvalidJson"),
    }
}

/// Parse a PowerShell command via the native AST parser (LRU-memoized).
///
/// Transient failures (spawn/timeout/JSON) are evicted so later calls retry;
/// deterministic results (valid parses, CommandTooLong) stay cached.
pub fn parse_powershell_command(command: &str) -> ParsedPowerShellCommand {
    {
        let mut cache = PARSE_CACHE.lock();
        if let Some(cached) = cache.map.get(command).cloned() {
            // move_to_end
            cache.order.retain(|k| k != command);
            cache.order.push_back(command.to_string());
            return cached;
        }
    }
    let result = parse_powershell_command_impl(command);
    let transient = !result.valid
        && result
            .errors
            .first()
            .map(|e| TRANSIENT_ERROR_IDS.contains(&e.error_id.as_str()))
            .unwrap_or(false);
    if !transient {
        let mut cache = PARSE_CACHE.lock();
        cache.map.insert(command.to_string(), result.clone());
        cache.order.retain(|k| k != command);
        cache.order.push_back(command.to_string());
        while cache.map.len() > PARSE_CACHE_MAX {
            if let Some(oldest) = cache.order.pop_front() {
                cache.map.remove(&oldest);
            } else {
                break;
            }
        }
    }
    result
}

/// Clear the parse cache (tests only).
pub fn reset_parse_cache() {
    let mut cache = PARSE_CACHE.lock();
    cache.map.clear();
    cache.order.clear();
}

// =============================================================================
// Analysis accessors — derived from the parsed AST (consumed by security /
// permission layers). Port of the parser.ts accessor section.
// =============================================================================

/// Common PowerShell aliases → canonical cmdlet (COMMON_ALIASES). Resolution is
/// single-hop. SECURITY: aliases that collide with native .exe on PS Core 6+
/// (sc/sort/curl/wget) are deliberately OMITTED — resolving them would
/// auto-allow the wrong program.
pub static COMMON_ALIASES: Lazy<HashMap<&'static str, &'static str>> = Lazy::new(|| {
    HashMap::from([
        // Directory listing
        ("ls", "Get-ChildItem"), ("dir", "Get-ChildItem"), ("gci", "Get-ChildItem"),
        // Content
        ("cat", "Get-Content"), ("type", "Get-Content"), ("gc", "Get-Content"),
        // Navigation
        ("cd", "Set-Location"), ("sl", "Set-Location"), ("chdir", "Set-Location"),
        ("pushd", "Push-Location"), ("popd", "Pop-Location"), ("pwd", "Get-Location"),
        ("gl", "Get-Location"),
        // Items
        ("gi", "Get-Item"), ("gp", "Get-ItemProperty"), ("ni", "New-Item"),
        ("mkdir", "New-Item"), ("md", "New-Item"),
        ("ri", "Remove-Item"), ("del", "Remove-Item"), ("rd", "Remove-Item"),
        ("rmdir", "Remove-Item"), ("rm", "Remove-Item"), ("erase", "Remove-Item"),
        ("mi", "Move-Item"), ("mv", "Move-Item"), ("move", "Move-Item"),
        ("ci", "Copy-Item"), ("cp", "Copy-Item"), ("copy", "Copy-Item"), ("cpi", "Copy-Item"),
        ("si", "Set-Item"), ("rni", "Rename-Item"), ("ren", "Rename-Item"),
        // Process
        ("ps", "Get-Process"), ("gps", "Get-Process"), ("kill", "Stop-Process"),
        ("spps", "Stop-Process"), ("start", "Start-Process"), ("saps", "Start-Process"),
        ("sajb", "Start-Job"), ("ipmo", "Import-Module"),
        // Output
        ("echo", "Write-Output"), ("write", "Write-Output"), ("sleep", "Start-Sleep"),
        // Help
        ("help", "Get-Help"), ("man", "Get-Help"), ("gcm", "Get-Command"),
        // Service
        ("gsv", "Get-Service"),
        // Variables
        ("gv", "Get-Variable"), ("sv", "Set-Variable"),
        // History
        ("h", "Get-History"), ("history", "Get-History"),
        // Invoke
        ("iex", "Invoke-Expression"), ("iwr", "Invoke-WebRequest"), ("irm", "Invoke-RestMethod"),
        ("icm", "Invoke-Command"), ("ii", "Invoke-Item"),
        // PSSession — remote code execution surface
        ("nsn", "New-PSSession"), ("etsn", "Enter-PSSession"), ("exsn", "Exit-PSSession"),
        ("gsn", "Get-PSSession"), ("rsn", "Remove-PSSession"),
        // Misc
        ("cls", "Clear-Host"), ("clear", "Clear-Host"), ("select", "Select-Object"),
        ("where", "Where-Object"), ("foreach", "ForEach-Object"), ("%", "ForEach-Object"),
        ("?", "Where-Object"),
        ("measure", "Measure-Object"), ("ft", "Format-Table"), ("fl", "Format-List"),
        ("fw", "Format-Wide"),
        ("oh", "Out-Host"), ("ogv", "Out-GridView"),
        ("ac", "Add-Content"), ("clc", "Clear-Content"), ("tee", "Tee-Object"),
        ("epcsv", "Export-Csv"),
        ("sp", "Set-ItemProperty"), ("rp", "Remove-ItemProperty"), ("cli", "Clear-Item"),
        ("epal", "Export-Alias"),
        // Text search
        ("sls", "Select-String"),
    ])
});

const DIRECTORY_CHANGE_CMDLETS: &[&str] = &["set-location", "push-location", "pop-location"];
const DIRECTORY_CHANGE_ALIASES: &[&str] = &["cd", "sl", "chdir", "pushd", "popd"];

/// Lowercased names across all statements + nested commands.
pub fn get_all_command_names(parsed: &ParsedPowerShellCommand) -> Vec<String> {
    let mut names = Vec::new();
    for statement in &parsed.statements {
        for cmd in &statement.commands {
            names.push(cmd.name.to_lowercase());
        }
        if let Some(nested) = &statement.nested_commands {
            for cmd in nested {
                names.push(cmd.name.to_lowercase());
            }
        }
    }
    names
}

/// Flat list of every command (statements + nested).
pub fn get_all_commands(parsed: &ParsedPowerShellCommand) -> Vec<&ParsedCommandElement> {
    let mut commands = Vec::new();
    for statement in &parsed.statements {
        commands.extend(statement.commands.iter());
        if let Some(nested) = &statement.nested_commands {
            commands.extend(nested.iter());
        }
    }
    commands
}

/// All redirections across statements + nested-command redirections.
pub fn get_all_redirections(parsed: &ParsedPowerShellCommand) -> Vec<&ParsedRedirection> {
    let mut redirections = Vec::new();
    for statement in &parsed.statements {
        redirections.extend(statement.redirections.iter());
        if let Some(nested) = &statement.nested_commands {
            for cmd in nested {
                if let Some(rs) = &cmd.redirections {
                    redirections.extend(rs.iter());
                }
            }
        }
    }
    redirections
}

/// Variables whose path starts with `<scope>:` (e.g. 'env').
pub fn get_variables_by_scope<'a>(
    parsed: &'a ParsedPowerShellCommand,
    scope: &str,
) -> Vec<&'a ParsedVariable> {
    let prefix = format!("{}:", scope.to_lowercase());
    parsed.variables.iter().filter(|v| v.path.to_lowercase().starts_with(&prefix)).collect()
}

/// Case-insensitive command presence, resolving aliases both directions.
pub fn has_command_named(parsed: &ParsedPowerShellCommand, name: &str) -> bool {
    let lower_name = name.to_lowercase();
    let canonical_from_alias =
        COMMON_ALIASES.get(lower_name.as_str()).map(|c| c.to_lowercase());
    for cmd_name in get_all_command_names(parsed) {
        if cmd_name == lower_name {
            return true;
        }
        let canonical = COMMON_ALIASES.get(cmd_name.as_str()).map(|c| c.to_lowercase());
        if canonical.as_deref() == Some(lower_name.as_str()) {
            return true;
        }
        if let Some(cfa) = &canonical_from_alias {
            if &cmd_name == cfa {
                return true;
            }
            if let Some(c) = &canonical {
                if c == cfa {
                    return true;
                }
            }
        }
    }
    false
}

/// True if any Set-/Push-/Pop-Location (or cd/sl/chdir/pushd/popd) present.
pub fn has_directory_change(parsed: &ParsedPowerShellCommand) -> bool {
    get_all_command_names(parsed).iter().any(|cmd_name| {
        DIRECTORY_CHANGE_CMDLETS.contains(&cmd_name.as_str())
            || DIRECTORY_CHANGE_ALIASES.contains(&cmd_name.as_str())
    })
}

/// True if one statement, one command, no nested commands.
pub fn is_single_command(parsed: &ParsedPowerShellCommand) -> bool {
    if parsed.statements.len() != 1 {
        return false;
    }
    let stmt = &parsed.statements[0];
    stmt.commands.len() == 1 && stmt.nested_commands.is_none()
}

/// Case-insensitive exact arg/flag presence.
pub fn command_has_arg(command: &ParsedCommandElement, arg: &str) -> bool {
    let lower = arg.to_lowercase();
    command.args.iter().any(|a| a.to_lowercase() == lower)
}

/// Tokenizer-level dash chars PowerShell accepts as parameter prefixes
/// (SpecialCharacters.IsDash): ASCII hyphen, en-dash, em-dash, horizontal bar.
pub const PS_TOKENIZER_DASH_CHARS: &[char] = &['-', '–', '—', '―'];

/// Rewrite a parameter token so it starts with an ASCII hyphen, dropping
/// whatever character stood there before.
///
/// This is Python's `f"-{arg[1:]}"`, which slices by CHARACTER. Doing it with
/// `&arg[1..]` in Rust slices by BYTE, and three of the four dash characters
/// above are multi-byte in UTF-8 — an en-dash occupies bytes 0..3, so byte
/// index 1 lands inside it and the slice panics. `is_powershell_parameter`
/// accepts those dashes (and any token the AST labelled "Parameter", including
/// an empty one), so `Get-ChildItem –Path .` used to abort the process from
/// inside the permission check. An empty token yields "-", matching what
/// Python's slice produces.
pub fn normalize_dash_param(arg: &str) -> String {
    let mut it = arg.char_indices();
    match it.next() {
        Some(_) => {
            let rest_start = it.next().map(|(i, _)| i).unwrap_or(arg.len());
            format!("-{}", &arg[rest_start..])
        }
        None => "-".to_string(),
    }
}

/// AST element type is authoritative ('Parameter'); else fall back to a
/// leading-dash-char check.
pub fn is_powershell_parameter(arg: &str, element_type: Option<&str>) -> bool {
    if let Some(et) = element_type {
        return et == "Parameter";
    }
    arg.chars().next().map(|c| PS_TOKENIZER_DASH_CHARS.contains(&c)).unwrap_or(false)
}

/// True if any arg is an unambiguous abbreviation of full_param (≥ min_prefix).
pub fn command_has_arg_abbreviation(
    command: &ParsedCommandElement,
    full_param: &str,
    min_prefix: &str,
) -> bool {
    let lower_full = full_param.to_lowercase();
    let lower_min = min_prefix.to_lowercase();
    for a in &command.args {
        // colon = a.find(":", 1): search from char index 1.
        let colon = a
            .char_indices()
            .skip(1)
            .find(|(_, c)| *c == ':')
            .map(|(i, _)| i);
        let param_part = match colon {
            Some(i) if i > 0 => &a[..i],
            _ => a.as_str(),
        };
        let lower = param_part.replace('`', "").to_lowercase();
        if lower.starts_with(&lower_min)
            && lower_full.starts_with(&lower)
            && lower.chars().count() <= lower_full.chars().count()
        {
            return true;
        }
    }
    false
}

/// Statements, each a pipeline segment for per-segment permission checks.
pub fn get_pipeline_segments(parsed: &ParsedPowerShellCommand) -> &[ParsedStatement] {
    &parsed.statements
}

/// True for `$null`/`${null}` — discards output, not a filesystem write.
pub fn is_null_redirection_target(target: &str) -> bool {
    let t = target.trim().to_lowercase();
    t == "$null" || t == "${null}"
}

/// File-writing redirections (excludes merging and $null sinks).
pub fn get_file_redirections(parsed: &ParsedPowerShellCommand) -> Vec<&ParsedRedirection> {
    get_all_redirections(parsed)
        .into_iter()
        .filter(|r| !r.is_merging && !is_null_redirection_target(&r.target))
        .collect()
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SecurityFlags {
    pub has_sub_expressions: bool,
    pub has_script_blocks: bool,
    pub has_splatting: bool,
    pub has_expandable_strings: bool,
    pub has_member_invocations: bool,
    pub has_assignments: bool,
    pub has_stop_parsing: bool,
}

/// Security-relevant flags from the parsed AST (deriveSecurityFlags).
pub fn derive_security_flags(parsed: &ParsedPowerShellCommand) -> SecurityFlags {
    let mut flags = SecurityFlags { has_stop_parsing: parsed.has_stop_parsing, ..Default::default() };

    let check_elements = |flags: &mut SecurityFlags, cmd: &ParsedCommandElement| {
        let Some(element_types) = &cmd.element_types else {
            return;
        };
        for et in element_types {
            match et.as_str() {
                "ScriptBlock" => flags.has_script_blocks = true,
                "SubExpression" => flags.has_sub_expressions = true,
                "ExpandableString" => flags.has_expandable_strings = true,
                "MemberInvocation" => flags.has_member_invocations = true,
                _ => {}
            }
        }
    };

    for stmt in &parsed.statements {
        if stmt.statement_type == "AssignmentStatementAst" {
            flags.has_assignments = true;
        }
        for cmd in &stmt.commands {
            check_elements(&mut flags, cmd);
        }
        if let Some(nested) = &stmt.nested_commands {
            for cmd in nested {
                check_elements(&mut flags, cmd);
            }
        }
        if let Some(sp) = &stmt.security_patterns {
            if sp.get("hasMemberInvocations").copied().unwrap_or(false) {
                flags.has_member_invocations = true;
            }
            if sp.get("hasSubExpressions").copied().unwrap_or(false) {
                flags.has_sub_expressions = true;
            }
            if sp.get("hasExpandableStrings").copied().unwrap_or(false) {
                flags.has_expandable_strings = true;
            }
            if sp.get("hasScriptBlocks").copied().unwrap_or(false) {
                flags.has_script_blocks = true;
            }
        }
    }

    for v in &parsed.variables {
        if v.is_splatted {
            flags.has_splatting = true;
            break;
        }
    }

    flags
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn budget_matches_python_derivation() {
        // Python: int(((((32767-200)*3/8) - len(script) - 21) * 3 / 4)) - 100
        let script_chars_budget = (32767.0 - 200.0) * 3.0 / 8.0;
        let cmd_b64 = script_chars_budget - PARSE_SCRIPT_BODY.len() as f64 - 21.0;
        let expect = ((cmd_b64 * 3.0 / 4.0) as i64 - 100).max(0) as usize;
        assert_eq!(*WINDOWS_MAX_COMMAND_LENGTH, expect);
        assert!(*WINDOWS_MAX_COMMAND_LENGTH > 0);
    }

    #[test]
    fn script_wrapper_shape() {
        let s = build_parse_script("dir");
        assert!(s.starts_with("$EncodedCommand = '"));
        // "dir" → ZGly
        assert!(s.starts_with("$EncodedCommand = 'ZGly'\n"));
    }

    #[test]
    fn utf16le_base64() {
        // "A" UTF-16LE = 41 00 → "QQA="
        assert_eq!(to_utf16le_base64("A"), "QQA=");
    }

    #[test]
    fn classify_and_strip() {
        assert_eq!(classify_command_name("Get-ChildItem"), "cmdlet");
        assert_eq!(classify_command_name("./run.sh"), "application");
        assert_eq!(classify_command_name("dir"), "unknown");
        assert_eq!(strip_module_prefix(r"Module\Get-Thing"), "Get-Thing");
        assert_eq!(strip_module_prefix(r"C:\x\y.exe"), r"C:\x\y.exe");
        assert_eq!(strip_module_prefix(r".\local.exe"), r".\local.exe");
        assert_eq!(strip_module_prefix(r"\\srv\share\x"), r"\\srv\share\x");
    }

    #[test]
    fn element_type_mapping() {
        assert_eq!(map_element_type("ScriptBlockExpressionAst", None), "ScriptBlock");
        assert_eq!(map_element_type("ArrayExpressionAst", None), "SubExpression");
        assert_eq!(map_element_type("ConstantExpressionAst", None), "StringConstant");
        assert_eq!(
            map_element_type("CommandExpressionAst", Some("SubExpressionAst")),
            "SubExpression"
        );
        assert_eq!(map_element_type("CommandExpressionAst", None), "Other");
        assert_eq!(map_element_type("WeirdAst", None), "Other");
    }

    #[test]
    fn transform_command_ast_quotes_and_nonascii() {
        // Quoted name is stripped; non-ASCII forces 'application'.
        let raw = json!({
            "text": "'Get-Item' x",
            "commandElements": [
                {"type": "StringConstantExpressionAst", "text": "'Get-Item'", "value": "Get-Item"},
                {"type": "StringConstantExpressionAst", "text": "x", "value": "x"}
            ]
        });
        let cmd = transform_command_ast(&raw);
        assert_eq!(cmd.name, "Get-Item");
        assert_eq!(cmd.name_type, "cmdlet");
        assert_eq!(cmd.args, vec!["x"]);

        let spoof = json!({
            "text": "Ｇet-Item",
            "commandElements": [
                {"type": "StringConstantExpressionAst", "text": "Ｇet-Item", "value": "Ｇet-Item"}
            ]
        });
        assert_eq!(transform_command_ast(&spoof).name_type, "application");
    }

    #[test]
    fn transform_redirections() {
        let merging = json!({"type": "MergingRedirectionAst"});
        let r = transform_redirection(&merging);
        assert!(r.is_merging);
        assert_eq!(r.operator, "2>&1");
        let err_append = json!({"type": "FileRedirectionAst", "append": true, "fromStream": "Error", "locationText": "log.txt"});
        let r = transform_redirection(&err_append);
        assert_eq!(r.operator, "2>>");
        assert_eq!(r.target, "log.txt");
        let plain = json!({"type": "FileRedirectionAst", "append": false, "fromStream": "Output", "locationText": "out.txt"});
        assert_eq!(transform_redirection(&plain).operator, ">");
    }

    #[test]
    fn ensure_array_unwrap() {
        // PS 5.1 single-element unwrap: an object where a list is expected.
        let single = json!({"type": "CommandAst"});
        assert_eq!(ensure_array(Some(&single)).len(), 1);
        assert_eq!(ensure_array(Some(&json!([1, 2]))).len(), 2);
        assert_eq!(ensure_array(None).len(), 0);
        assert_eq!(ensure_array(Some(&Value::Null)).len(), 0);
    }

    #[test]
    fn alias_resolution_both_directions() {
        let parsed = ParsedPowerShellCommand {
            valid: true,
            statements: vec![ParsedStatement {
                statement_type: "PipelineAst".into(),
                commands: vec![ParsedCommandElement {
                    name: "ls".into(),
                    name_type: "unknown".into(),
                    element_type: "CommandAst".into(),
                    ..Default::default()
                }],
                ..Default::default()
            }],
            ..Default::default()
        };
        assert!(has_command_named(&parsed, "Get-ChildItem")); // alias → canonical
        assert!(has_command_named(&parsed, "ls")); // exact
        assert!(has_command_named(&parsed, "dir")); // alias ↔ alias via canonical
        assert!(!has_command_named(&parsed, "Remove-Item"));
        assert!(has_directory_change(&ParsedPowerShellCommand {
            statements: vec![ParsedStatement {
                commands: vec![ParsedCommandElement { name: "cd".into(), ..Default::default() }],
                ..Default::default()
            }],
            ..Default::default()
        }));
    }

    #[test]
    fn parameter_and_abbreviation() {
        assert!(is_powershell_parameter("-Force", None));
        assert!(is_powershell_parameter("–Force", None)); // en-dash
        assert!(!is_powershell_parameter("Force", None));
        assert!(is_powershell_parameter("x", Some("Parameter")));
        assert!(!is_powershell_parameter("-x", Some("StringConstant")));

        let cmd = ParsedCommandElement {
            args: vec!["-Recur`se".into()],
            ..Default::default()
        };
        assert!(command_has_arg_abbreviation(&cmd, "-Recurse", "-Rec"));
        let cmd2 = ParsedCommandElement { args: vec!["-Re".into()], ..Default::default() };
        assert!(!command_has_arg_abbreviation(&cmd2, "-Recurse", "-Rec")); // below min
        let cmd3 = ParsedCommandElement { args: vec!["-Recurse:$true".into()], ..Default::default() };
        assert!(command_has_arg_abbreviation(&cmd3, "-Recurse", "-Rec")); // colon-split
    }

    #[test]
    fn null_redirection() {
        assert!(is_null_redirection_target("$null"));
        assert!(is_null_redirection_target(" ${NULL} "));
        assert!(!is_null_redirection_target("file.txt"));
    }

    #[test]
    fn security_flags_from_patterns() {
        let parsed = ParsedPowerShellCommand {
            statements: vec![ParsedStatement {
                statement_type: "AssignmentStatementAst".into(),
                security_patterns: Some(HashMap::from([
                    ("hasScriptBlocks".to_string(), true),
                    ("hasSubExpressions".to_string(), true),
                ])),
                ..Default::default()
            }],
            variables: vec![ParsedVariable { path: "args".into(), is_splatted: true }],
            has_stop_parsing: true,
            ..Default::default()
        };
        let f = derive_security_flags(&parsed);
        assert!(f.has_assignments && f.has_script_blocks && f.has_sub_expressions);
        assert!(f.has_splatting && f.has_stop_parsing);
        assert!(!f.has_member_invocations);
    }

    #[test]
    fn transform_statement_dedups_deep_redirections() {
        let raw = json!({
            "type": "PipelineAst",
            "text": "cmd > out.txt",
            "elements": [{
                "type": "CommandAst",
                "text": "cmd > out.txt",
                "commandElements": [{"type": "StringConstantExpressionAst", "text": "cmd", "value": "cmd"}],
                "redirections": [{"type": "FileRedirectionAst", "append": false, "fromStream": "Output", "locationText": "out.txt"}]
            }],
            "redirections": [{"type": "FileRedirectionAst", "append": false, "fromStream": "Output", "locationText": "out.txt"}]
        });
        let stmt = transform_statement(&raw);
        assert_eq!(stmt.redirections.len(), 1); // deep FindAll copy deduped
    }

    /// Three of the four dashes PowerShell accepts are multi-byte, so the old
    /// `&arg[1..]` sliced inside the character and panicked. Every dash must
    /// normalize to an ASCII hyphen with the rest of the token intact.
    #[test]
    fn dash_param_normalizes_every_unicode_dash() {
        for dash in PS_TOKENIZER_DASH_CHARS {
            let arg = format!("{dash}Path");
            assert!(
                is_powershell_parameter(&arg, None),
                "{arg:?} should read as a parameter"
            );
            assert_eq!(normalize_dash_param(&arg), "-Path", "failed for {dash:?}");
        }
        // A token the AST labelled "Parameter" need not start with a dash, and
        // it can be empty; neither may panic.
        assert_eq!(normalize_dash_param(""), "-");
        assert_eq!(normalize_dash_param("Path"), "-ath");
        // Multi-byte content after the dash survives untouched.
        assert_eq!(normalize_dash_param("–경로"), "-경로");
    }
}
