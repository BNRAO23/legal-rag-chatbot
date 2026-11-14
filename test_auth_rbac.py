"""Automated verification suite for Authentication, RBAC, and Daily Quotas.

Tests three critical scenarios:
1. Not signed in: Unauthenticated access attempt is blocked.
2. Wrong role: User with role 'tax_consultant' asks a Companies Act question ->
   role check before search blocks unauthorized document access -> 0 chunks returned ->
   refusal response, while user with role 'corporate_lawyer' succeeds.
3. Over the cap: User hits their daily quota -> clear message displayed, further
   questions rejected before search.
"""

from __future__ import annotations

import datetime
import os
import sys
import time
from typing import Any

from answer import answer
from auth import (
    authenticate_user,
    get_allowed_documents,
    get_role_description,
    load_auth_config,
    usage_tracker,
)
from retrieve import retrieve


def print_banner(title: str) -> None:
    print("=" * 80)
    print(title)
    print("=" * 80)


def test_scenario_1_not_signed_in() -> bool:
    print_banner("SCENARIO 1: Testing 'Not Signed In' Behavior")
    print("Requirement: Nobody can ask a question without signing in.")
    print("-" * 80)

    # 1. Attempt authentication with missing / empty credentials
    res_none = authenticate_user("", "")
    res_bad = authenticate_user("unknown_user", "wrong_password")
    res_bad_pw = authenticate_user("corp_user", "invalid_secret_pass")

    print(f"Empty credentials test   : {res_none} (Expected: None)")
    print(f"Unknown username test    : {res_bad} (Expected: None)")
    print(f"Wrong password test      : {res_bad_pw} (Expected: None)")

    auth_blocked = (res_none is None) and (res_bad is None) and (res_bad_pw is None)

    # 2. Verify that an unauthenticated user has 0 allowed documents
    unauth_docs = get_allowed_documents("unauthenticated")
    print(f"Unauthenticated role docs: {unauth_docs} (Expected: [])")

    # 3. Verify retrieval with empty allowed_documents returns 0 chunks
    unauth_retrieval = retrieve(
        "What is the quorum under Section 174?",
        allowed_documents=unauth_docs,
    )
    print(f"Unauthenticated retrieval: {len(unauth_retrieval)} chunks returned (Expected: 0)")

    passed = auth_blocked and (len(unauth_docs) == 0) and (len(unauth_retrieval) == 0)
    print(f"\nScenario 1 Result: {'PASS ✅' if passed else 'FAIL ❌'}")
    print()
    return passed


def test_scenario_2_wrong_role() -> bool:
    print_banner("SCENARIO 2: Testing 'Wrong Role' Document Access Control")
    print("Requirement: Check the role before the search so a user only gets chunks from documents that role may see.")
    print("-" * 80)

    # Authenticate tax user (role: tax_consultant)
    tax_auth = authenticate_user("tax_user", "tax_password_2026")
    assert tax_auth is not None, "Failed to authenticate tax_user from secrets.toml"
    tax_role = tax_auth["role"]
    tax_allowed = get_allowed_documents(tax_role)
    print(f"User: @{tax_auth['username']} | Role: {tax_role} ({get_role_description(tax_role)})")
    print(f"Authorized Documents for {tax_role}: {tax_allowed}")
    print()

    # Authenticate corporate user (role: corporate_lawyer)
    corp_auth = authenticate_user("corp_user", "corp_password_2026")
    assert corp_auth is not None, "Failed to authenticate corp_user from secrets.toml"
    corp_role = corp_auth["role"]
    corp_allowed = get_allowed_documents(corp_role)
    print(f"User: @{corp_auth['username']} | Role: {corp_role} ({get_role_description(corp_role)})")
    print(f"Authorized Documents for {corp_role}: {corp_allowed}")
    print("-" * 80)

    question = "Explain the restrictions on loans to directors under Section 185 of the Companies Act, 2013."
    print(f"Query: \"{question}\"")
    print()

    # Step A: Tax Consultant attempts to query Companies Act
    print("[Test 2A] Tax Consultant (Wrong Role for Companies Act):")
    t0 = time.perf_counter()
    tax_chunks = retrieve(question, allowed_documents=tax_allowed)
    print(f"  Role check before search: Target 'Companies Act, 2013' vs Allowed {tax_allowed}")
    print(f"  Retrieved chunks count  : {len(tax_chunks)} (Expected: 0, unauthorized document blocked)")

    tax_answer = answer(question, allowed_documents=tax_allowed, guardrails=False)
    print(f"  Assistant Answer        : \"{tax_answer['answer']}\"")
    print(f"  Latency                 : {time.perf_counter() - t0:.2f}s")
    print()

    # Step B: Corporate Lawyer queries Companies Act (Authorized Role)
    print("[Test 2B] Corporate Lawyer (Authorized Role for Companies Act):")
    t1 = time.perf_counter()
    corp_chunks = retrieve(question, allowed_documents=corp_allowed)
    print(f"  Role check before search: Target 'Companies Act, 2013' vs Allowed {corp_allowed}")
    print(f"  Retrieved chunks count  : {len(corp_chunks)} (Expected: >= 1 chunks from Companies Act)")
    if corp_chunks:
        print(f"  Top Chunk Source        : {corp_chunks[0]['metadata'].get('document_name')} | Section {corp_chunks[0]['metadata'].get('unit_number')}")

    corp_answer = answer(question, allowed_documents=corp_allowed, guardrails=False)
    print(f"  Assistant Answer        :\n  {corp_answer['answer'][:200]}...")
    print(f"  Latency                 : {time.perf_counter() - t1:.2f}s")

    # Assertions
    passed_tax_blocked = (len(tax_chunks) == 0) and ("could not find this in the documents" in tax_answer["answer"])
    passed_corp_allowed = (len(corp_chunks) > 0) and ("could not find this in the documents" not in corp_answer["answer"])
    passed = passed_tax_blocked and passed_corp_allowed

    print(f"\nScenario 2 Result: {'PASS ✅' if passed else 'FAIL ❌'}")
    print()
    return passed


def test_scenario_3_over_the_cap() -> bool:
    print_banner("SCENARIO 3: Testing 'Over the Cap' Daily Question Limit")
    print("Requirement: Add a daily cap on questions per user with a clear message when the cap is reached.")
    print("-" * 80)

    # Use capped_user (daily_limit = 2)
    user_auth = authenticate_user("capped_user", "intern_password_2026")
    assert user_auth is not None, "Failed to authenticate capped_user from secrets.toml"
    username = user_auth["username"]
    limit = user_auth["daily_limit"]

    # Reset usage for a clean test
    usage_tracker.reset_usage(username)
    print(f"User: @{username} | Daily Quota Limit: {limit} questions/day")
    print(f"Initial Usage for Today: {usage_tracker.get_usage(username)}")
    print("-" * 80)

    # Question 1 (Should be ALLOWED)
    ok_1, used_1, _, msg_1 = usage_tracker.check_daily_cap(username, limit)
    print(f"[Question 1] Quota Check: {'ALLOWED ✅' if ok_1 else 'BLOCKED ❌'} (Used: {used_1}/{limit})")
    assert ok_1, "Question 1 should be allowed"
    usage_tracker.increment_usage(username)

    # Question 2 (Should be ALLOWED)
    ok_2, used_2, _, msg_2 = usage_tracker.check_daily_cap(username, limit)
    print(f"[Question 2] Quota Check: {'ALLOWED ✅' if ok_2 else 'BLOCKED ❌'} (Used: {used_2}/{limit})")
    assert ok_2, "Question 2 should be allowed"
    usage_tracker.increment_usage(username)

    # Question 3 (Should be BLOCKED - Over the Cap)
    ok_3, used_3, _, cap_error_msg = usage_tracker.check_daily_cap(username, limit)
    print(f"[Question 3] Quota Check: {'ALLOWED ✅' if ok_3 else 'BLOCKED 🛡️ (CAP REACHED)'} (Used: {used_3}/{limit})")
    print(f"Clear Rejection Message :\n\"{cap_error_msg}\"")

    passed = (ok_1 is True) and (ok_2 is True) and (ok_3 is False) and ("Daily question limit reached" in cap_error_msg)
    print(f"\nScenario 3 Result: {'PASS ✅' if passed else 'FAIL ❌'}")
    print()
    return passed


def main() -> None:
    print("=" * 80)
    print("LEGAL RAG CHATBOT: AUTHENTICATION, RBAC & DAILY CAP VERIFICATION")
    print("=" * 80)
    print(f"Date: {datetime.date.today().isoformat()}")

    # Verify secrets file presence
    users, roles = load_auth_config()
    print(f"Configured Users in Secrets : {list(users.keys())}")
    print(f"Configured Roles in Secrets : {list(roles.keys())}")
    print()

    p1 = test_scenario_1_not_signed_in()
    p2 = test_scenario_2_wrong_role()
    p3 = test_scenario_3_over_the_cap()

    print_banner("SUMMARY SCORECARD")
    print(f"Test 1 (Not Signed In) : {'PASS ✅' if p1 else 'FAIL ❌'}")
    print(f"Test 2 (Wrong Role)    : {'PASS ✅' if p2 else 'FAIL ❌'}")
    print(f"Test 3 (Over the Cap)  : {'PASS ✅' if p3 else 'FAIL ❌'}")
    print("=" * 80)

    if p1 and p2 and p3:
        print("All 3 authentication and RBAC tests PASSED successfully!")
    else:
        print("One or more tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()

